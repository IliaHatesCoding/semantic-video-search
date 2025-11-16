import os
import sys
import logging
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import psycopg2
from psycopg2 import OperationalError
from datetime import datetime

# Load environment variables
load_dotenv()

# Configuration
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('inference.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def get_db_connection():
    """Get PostgreSQL connection from DATABASE_URL environment variable."""
    database_url = 'postgresql://postgres:cBQYxYIlOrIWfRtLTIcMFWcvenRPBjzX@yamabiko.proxy.rlwy.net:42256/railway'

    try:
        conn = psycopg2.connect(database_url)
        logger.info("Successfully connected to PostgreSQL database")
        return conn
    except OperationalError as e:
        logger.error(f"Database connection error: {e}")
        logger.error("Please check your DATABASE_URL and ensure the database is accessible")
        raise
    except Exception as e:
        logger.error(f"Error connecting to database: {e}")
        raise


def encode_query(model, query):
    """Encode user query to embedding vector."""
    try:
        embedding = model.encode(query, convert_to_numpy=True)
        return embedding.tolist()
    except Exception as e:
        logger.error(f"Error encoding query: {e}")
        raise


def search_similar_segments(conn, query_embedding, num_results=100, min_similarity=0.4):
    """Search for similar segments using cosine similarity.
    
    Args:
        conn: Database connection
        query_embedding: Query embedding vector
        num_results: Maximum number of results to fetch (before filtering)
        min_similarity: Minimum similarity threshold (0.4 = 40%)
    
    Returns:
        List of result dictionaries filtered by similarity threshold
    """
    try:
        with conn.cursor() as cur:
            # Format embedding as string for pgvector
            embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
            
            # Use cosine distance (1 - cosine similarity) for similarity search
            # Order by distance ascending (most similar first)
            # Fetch more results to account for filtering
            search_query = """
            SELECT 
                video_id,
                title,
                description,
                published_at,
                duration,
                view_count,
                like_count,
                comment_count,
                favorite_count,
                transcription_language,
                segment_start,
                segment_end,
                segment_text,
                1 - (embedding <=> %s::vector) as similarity
            FROM transcriptions
            WHERE embedding IS NOT NULL
            AND (1 - (embedding <=> %s::vector)) >= %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
            """
            
            cur.execute(search_query, (embedding_str, embedding_str, min_similarity, embedding_str, num_results))
            results = cur.fetchall()
            
            # Convert to list of dictionaries
            columns = [
                'video_id', 'title', 'description', 'published_at', 'duration',
                'view_count', 'like_count', 'comment_count', 'favorite_count',
                'transcription_language', 'segment_start', 'segment_end', 'segment_text', 'similarity'
            ]
            
            results_list = []
            for row in results:
                result_dict = dict(zip(columns, row))
                # Double-check similarity threshold (in case of floating point issues)
                if result_dict['similarity'] and result_dict['similarity'] >= min_similarity:
                    results_list.append(result_dict)
            
            logger.info(f"Found {len(results_list)} similar segments (similarity >= {min_similarity*100:.0f}%)")
            return results_list
            
    except Exception as e:
        logger.error(f"Error searching segments: {e}")
        raise


def group_results_by_video(results):
    """Group results by unique video_id, keeping segments sorted by similarity."""
    video_groups = {}
    
    for result in results:
        video_id = result['video_id']
        if video_id not in video_groups:
            video_groups[video_id] = {
                'video_info': {
                    'video_id': video_id,
                    'title': result['title'],
                    'description': result['description'],
                    'published_at': result['published_at'],
                    'duration': result['duration'],
                    'view_count': result['view_count'],
                    'like_count': result['like_count'],
                    'comment_count': result['comment_count'],
                    'favorite_count': result['favorite_count'],
                },
                'segments': []
            }
        video_groups[video_id]['segments'].append(result)
    
    # Sort segments within each video by similarity (highest first)
    for video_id in video_groups:
        video_groups[video_id]['segments'].sort(key=lambda x: x['similarity'] or 0, reverse=True)
    
    # Sort videos by their best segment's similarity
    sorted_videos = sorted(
        video_groups.items(),
        key=lambda x: x[1]['segments'][0]['similarity'] or 0,
        reverse=True
    )
    
    return dict(sorted_videos)


def format_duration(seconds):
    """Format seconds to MM:SS or HH:MM:SS format."""
    if seconds is None:
        return "0:00"
    
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes}:{secs:02d}"


def format_number(num):
    """Format large numbers with K, M suffixes."""
    if num is None:
        return "0"
    if num >= 1_000_000:
        return f"{num / 1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.1f}K"
    else:
        return str(num)


def generate_html(video_groups, query):
    """Generate HTML page with video results grouped by video with expandable segments."""
    
    # Count total segments
    total_segments = sum(len(group['segments']) for group in video_groups.values())
    unique_videos = len(video_groups)
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="referrer" content="strict-origin-when-cross-origin">
    <title>Video Search Results: "{query}"</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        .header {{
            background: white;
            border-radius: 12px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        
        .header h1 {{
            color: #333;
            font-size: 28px;
            margin-bottom: 10px;
        }}
        
        .query {{
            color: #667eea;
            font-size: 18px;
            font-weight: 600;
        }}
        
        .results-count {{
            color: #666;
            font-size: 14px;
            margin-top: 10px;
        }}
        
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
            gap: 25px;
            margin-bottom: 30px;
        }}
        
        .card {{
            background: white;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}
        
        .card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 8px 15px rgba(0, 0, 0, 0.2);
        }}
        
        .video-container {{
            position: relative;
            width: 100%;
            padding-bottom: 56.25%; /* 16:9 aspect ratio */
            background: #000;
        }}
        
        .video-container iframe {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            border: none;
        }}
        
        .card-content {{
            padding: 20px;
        }}
        
        .expand-button {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            margin-top: 15px;
            padding: 10px 20px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 6px;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.3s ease;
            font-size: 14px;
        }}
        
        .expand-button:hover {{
            background: #5568d3;
        }}
        
        .expand-button .icon {{
            transition: transform 0.3s ease;
        }}
        
        .expand-button.expanded .icon {{
            transform: rotate(180deg);
        }}
        
        .segments-list {{
            display: none;
            margin-top: 20px;
            padding-top: 20px;
            border-top: 2px solid #f0f0f0;
        }}
        
        .segments-list.expanded {{
            display: block;
        }}
        
        .segment-item {{
            background: #f8f9fa;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
            border-left: 4px solid #667eea;
        }}
        
        .segment-item:last-child {{
            margin-bottom: 0;
        }}
        
        .segment-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
            flex-wrap: wrap;
            gap: 10px;
        }}
        
        .segment-text-small {{
            color: #666;
            font-size: 13px;
            line-height: 1.5;
            margin-bottom: 10px;
        }}
        
        .segment-link {{
            display: inline-block;
            padding: 6px 12px;
            background: #667eea;
            color: white;
            text-decoration: none;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 600;
            transition: background 0.3s ease;
        }}
        
        .segment-link:hover {{
            background: #5568d3;
        }}
        
        .segments-count {{
            color: #667eea;
            font-weight: 600;
            font-size: 14px;
        }}
        
        .card-title {{
            font-size: 18px;
            font-weight: 600;
            color: #333;
            margin-bottom: 12px;
            line-height: 1.4;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
            overflow: hidden;
        }}
        
        .card-title a {{
            color: #667eea;
            text-decoration: none;
        }}
        
        .card-title a:hover {{
            text-decoration: underline;
        }}
        
        .segment-text {{
            color: #666;
            font-size: 14px;
            line-height: 1.6;
            margin-bottom: 15px;
            display: -webkit-box;
            -webkit-line-clamp: 3;
            -webkit-box-orient: vertical;
            overflow: hidden;
        }}
        
        .segment-info {{
            display: flex;
            align-items: center;
            gap: 15px;
            margin-bottom: 12px;
            flex-wrap: wrap;
        }}
        
        .time-badge {{
            background: #667eea;
            color: white;
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 600;
        }}
        
        .similarity-badge {{
            background: #10b981;
            color: white;
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 600;
        }}
        
        .stats {{
            display: flex;
            gap: 15px;
            font-size: 12px;
            color: #999;
            flex-wrap: wrap;
        }}
        
        .stat-item {{
            display: flex;
            align-items: center;
            gap: 5px;
        }}
        
        .youtube-link {{
            display: inline-block;
            margin-top: 15px;
            padding: 10px 20px;
            background: #ff0000;
            color: white;
            text-decoration: none;
            border-radius: 6px;
            font-weight: 600;
            transition: background 0.3s ease;
        }}
        
        .youtube-link:hover {{
            background: #cc0000;
        }}
        
        .footer {{
            text-align: center;
            color: white;
            padding: 20px;
            font-size: 14px;
        }}
        
        @media (max-width: 768px) {{
            .grid {{
                grid-template-columns: 1fr;
            }}
            
            .header {{
                padding: 20px;
            }}
            
            .header h1 {{
                font-size: 24px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎬 Video Search Results</h1>
            <div class="query">Query: "{query}"</div>
            <div class="results-count">Found {unique_videos} unique videos ({total_segments} segments total, similarity ≥ 40%)</div>
        </div>
        
        <div class="grid">
"""
    
    for video_idx, (video_id, video_data) in enumerate(video_groups.items(), 1):
        video_info = video_data['video_info']
        segments = video_data['segments']
        
        # Get the best segment (first one, already sorted by similarity)
        best_segment = segments[0]
        
        title = video_info['title'] or 'Untitled Video'
        view_count = video_info['view_count'] or 0
        like_count = video_info['like_count'] or 0
        
        # Best segment details
        best_segment_text = best_segment['segment_text'] or ''
        best_segment_start = best_segment['segment_start'] or 0
        best_segment_end = best_segment['segment_end'] or 0
        best_similarity = best_segment['similarity'] or 0
        best_similarity_pct = f"{best_similarity * 100:.1f}%"
        
        # YouTube embed URL for best segment
        start_seconds = max(0, int(best_segment_start))
        embed_url = f"https://www.youtube.com/embed/{video_id}?start={start_seconds}"
        watch_url = f"https://www.youtube.com/watch?v={video_id}&t={start_seconds}s"
        time_range = f"{format_duration(best_segment_start)} - {format_duration(best_segment_end)}"
        
        # Count additional segments
        additional_segments_count = len(segments) - 1
        
        html_content += f"""
            <div class="card">
                <div class="video-container">
                    <iframe 
                        id="ytplayer-{video_idx}"
                        type="text/html"
                        width="100%"
                        height="100%"
                        src="{embed_url}"
                        frameborder="0"
                        referrerpolicy="strict-origin-when-cross-origin"
                        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                        allowfullscreen>
                    </iframe>
                </div>
                <div class="card-content">
                    <div class="card-title">
                        <a href="{watch_url}" target="_blank">{title}</a>
                    </div>
                    <div class="segment-text">{best_segment_text}</div>
                    <div class="segment-info">
                        <span class="time-badge">⏱️ {time_range}</span>
                        <span class="similarity-badge">🎯 {best_similarity_pct} match</span>
                    </div>
                    <div class="stats">
                        <span class="stat-item">👁️ {format_number(view_count)} views</span>
                        <span class="stat-item">👍 {format_number(like_count)} likes</span>
                    </div>
                    <a href="{watch_url}" target="_blank" class="youtube-link">
                        ▶️ Watch on YouTube
                    </a>
"""
        
        # Add expand button if there are additional segments
        if additional_segments_count > 0:
            html_content += f"""
                    <button class="expand-button" onclick="toggleSegments('segments-{video_idx}')">
                        <span class="icon">▼</span>
                        <span>Show {additional_segments_count} more segment{'' if additional_segments_count == 1 else 's'}</span>
                    </button>
                    <div id="segments-{video_idx}" class="segments-list">
"""
            
            # Add all additional segments
            for seg_idx, segment in enumerate(segments[1:], 2):
                seg_text = segment['segment_text'] or ''
                seg_start = segment['segment_start'] or 0
                seg_end = segment['segment_end'] or 0
                seg_similarity = segment['similarity'] or 0
                seg_similarity_pct = f"{seg_similarity * 100:.1f}%"
                seg_start_seconds = max(0, int(seg_start))
                seg_watch_url = f"https://www.youtube.com/watch?v={video_id}&t={seg_start_seconds}s"
                seg_time_range = f"{format_duration(seg_start)} - {format_duration(seg_end)}"
                
                html_content += f"""
                        <div class="segment-item">
                            <div class="segment-header">
                                <span class="time-badge">⏱️ {seg_time_range}</span>
                                <span class="similarity-badge">🎯 {seg_similarity_pct} match</span>
                            </div>
                            <div class="segment-text-small">{seg_text}</div>
                            <a href="{seg_watch_url}" target="_blank" class="segment-link">
                                ▶️ Watch this segment
                            </a>
                        </div>
"""
            
            html_content += """
                    </div>
"""
        
        html_content += """
                </div>
            </div>
"""
    
    html_content += """
        </div>
        <div class="footer">
            Generated on """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """
        </div>
    </div>
    <script>
        function toggleSegments(segmentId) {{
            const segmentsList = document.getElementById(segmentId);
            const button = segmentsList.previousElementSibling;
            
            if (segmentsList.classList.contains('expanded')) {{
                segmentsList.classList.remove('expanded');
                button.classList.remove('expanded');
                const count = segmentsList.children.length;
                button.innerHTML = '<span class="icon">▼</span><span>Show ' + count + ' more segment' + (count === 1 ? '' : 's') + '</span>';
            }} else {{
                segmentsList.classList.add('expanded');
                button.classList.add('expanded');
                button.innerHTML = '<span class="icon">▼</span><span>Hide segments</span>';
            }}
        }}
    </script>
</body>
</html>
"""
    
    return html_content


def main():
    """Main function to perform semantic search and generate HTML."""
    # Get query from command line or user input
    if len(sys.argv) > 1:
        query = ' '.join(sys.argv[1:])
    else:
        query = input("Enter your search query: ").strip()
    
    if not query:
        logger.error("Query cannot be empty")
        return
    
    logger.info("="*60)
    logger.info(f"Starting semantic search for query: '{query}'")
    logger.info("="*60)
    
    # Load embedding model
    logger.info(f"Loading embedding model: {MODEL_NAME}")
    try:
        model = SentenceTransformer(MODEL_NAME)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return
    
    # Connect to database
    try:
        conn = get_db_connection()
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        return
    
    try:
        # Encode query
        logger.info("Encoding query...")
        query_embedding = encode_query(model, query)
        
        # Search for similar segments (fetch more to account for filtering and grouping)
        logger.info(f"Searching for similar segments (minimum similarity: 40%)...")
        results = search_similar_segments(conn, query_embedding, num_results=200, min_similarity=0.4)
        
        if not results:
            logger.warning("No results found with similarity >= 40%")
            print("No results found for your query (minimum similarity: 40%).")
            return
        
        # Group results by unique videos
        logger.info("Grouping results by unique videos...")
        video_groups = group_results_by_video(results)
        
        if not video_groups:
            logger.warning("No video groups found")
            print("No video groups found.")
            return
        
        logger.info(f"Found {len(video_groups)} unique videos with {len(results)} total segments")
        
        # Generate HTML
        logger.info("Generating HTML page...")
        html_content = generate_html(video_groups, query)
        
        # Save HTML file
        output_file = f"search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML page saved to: {output_file}")
        print(f"\n✅ Search complete! Results saved to: {output_file}")
        print(f"📊 Found {len(results)} results")
        print(f"🌐 Open {output_file} in your browser to view the results")
        
    except Exception as e:
        logger.error(f"Error in main process: {e}")
    finally:
        conn.close()
        logger.info("Database connection closed")


if __name__ == "__main__":
    main()

