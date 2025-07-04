import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional
import re
import string
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

class PlagiarismDetector:
    def __init__(self):
        self.models: Dict[str, Optional[SentenceTransformer]] = {
            'all-MiniLM-L6-v2': None,
            'all-mpnet-base-v2': None,
            'paraphrase-multilingual-MiniLM-L12-v2': None
        }
        self.similarity_threshold = 0.8
        
    def load_model(self, model_name: str):
        """Load a sentence transformer model"""
        if self.models[model_name] is None:
            try:
                self.models[model_name] = SentenceTransformer(model_name)
                return True
            except Exception as e:
                st.error(f"Error loading model {model_name}: {str(e)}")
                return False
        return True
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove special characters but keep punctuation for context
        text = re.sub(r'[^\w\s\.\,\!\?\;\:]', '', text)
        
        # Convert to lowercase
        text = text.lower()
        
        return text
    
    def generate_embeddings(self, texts: List[str], model_name: str) -> Optional[np.ndarray]:
        """Generate embeddings for a list of texts"""
        if not self.load_model(model_name):
            return None
            
        preprocessed_texts = [self.preprocess_text(text) for text in texts]
        
        model = self.models[model_name]
        if model is None :
            return None
        try:
            embeddings =  model.encode(preprocessed_texts)
            return embeddings
        except Exception as e:
            st.error(f"Error generating embeddings: {str(e)}")
            return None
    
    def calculate_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """Calculate pairwise cosine similarity matrix"""
        similarity_matrix = cosine_similarity(embeddings)
        return similarity_matrix
    
    def detect_clones(self, similarity_matrix: np.ndarray, texts: List[str]) -> List[Dict]:
        """Detect potential clones based on similarity threshold"""
        clones = []
        n_texts = len(texts)
        
        for i in range(n_texts):
            for j in range(i + 1, n_texts):
                similarity = similarity_matrix[i][j]
                if similarity >= self.similarity_threshold:
                    clones.append({
                        'text1_idx': i,
                        'text2_idx': j,
                        'text1_preview': texts[i][:100] + "..." if len(texts[i]) > 100 else texts[i],
                        'text2_preview': texts[j][:100] + "..." if len(texts[j]) > 100 else texts[j],
                        'similarity': similarity
                    })
        
        return sorted(clones, key=lambda x: x['similarity'], reverse=True)
    
    def create_similarity_heatmap(self, similarity_matrix: np.ndarray, text_labels: List[str]) -> go.Figure:
        """Create an interactive heatmap of similarity scores"""
        fig = go.Figure(data=go.Heatmap(
            z=similarity_matrix,
            x=text_labels,
            y=text_labels,
            colorscale='RdYlBu_r',
            zmin=0,
            zmax=1,
            text=np.round(similarity_matrix, 3),
            texttemplate="%{text}",
            textfont={"size": 10},
            colorbar=dict(title="Similarity Score")
        ))
        
        fig.update_layout(
            title="Text Similarity Matrix",
            xaxis_title="Texts",
            yaxis_title="Texts",
            width=700,
            height=600
        )
        
        return fig
    
    def compare_models(self, texts: List[str]) -> Dict:
        """Compare performance of different embedding models"""
        results = {}
        
        for model_name in self.models.keys():
            st.write(f"Processing with {model_name}...")
            
            embeddings = self.generate_embeddings(texts, model_name)
            if embeddings is not None:
                similarity_matrix = self.calculate_similarity_matrix(embeddings)
                clones = self.detect_clones(similarity_matrix, texts)
                
                results[model_name] = {
                    'embeddings': embeddings,
                    'similarity_matrix': similarity_matrix,
                    'clones': clones,
                    'avg_similarity': np.mean(similarity_matrix[np.triu_indices(len(texts), k=1)])
                }
        
        return results

def main():
    st.set_page_config(
        page_title="Plagiarism Detector",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Plagiarism Detector - Semantic Similarity Analyzer")
    st.markdown("Upload multiple texts to detect potential plagiarism using advanced semantic similarity analysis.")
    
    # Initialize detector
    detector = PlagiarismDetector()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Similarity threshold
        threshold = st.slider(
            "Similarity Threshold for Clone Detection",
            min_value=0.5,
            max_value=1.0,
            value=0.8,
            step=0.05,
            help="Texts with similarity above this threshold will be flagged as potential clones"
        )
        detector.similarity_threshold = threshold
        
        # Model selection
        selected_models = st.multiselect(
            "Select Embedding Models",
            options=list(detector.models.keys()),
            default=['all-MiniLM-L6-v2'],
            help="Choose which models to use for comparison"
        )
        
        st.info("💡 **Model Info:**\n- **all-MiniLM-L6-v2**: Fast, good for general use\n- **all-mpnet-base-v2**: Best quality, slower\n- **paraphrase-multilingual**: Good for multiple languages")
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 Input Texts")
        
        # Dynamic text input
        if 'num_texts' not in st.session_state:
            st.session_state.num_texts = 3
        
        num_texts = st.number_input(
            "Number of texts to compare",
            min_value=2,
            max_value=10,
            value=st.session_state.num_texts,
            step=1
        )
        st.session_state.num_texts = num_texts
        
        texts = []
        for i in range(num_texts):
            text = st.text_area(
                f"Text {i+1}",
                height=100,
                placeholder=f"Enter text {i+1} here...",
                key=f"text_{i}"
            )
            if text.strip():
                texts.append(text.strip())
        
        # File upload option
        st.subheader("📁 Or Upload Files")
        uploaded_files = st.file_uploader(
            "Upload text files",
            type=['txt'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            for file in uploaded_files:
                content = file.read().decode('utf-8')
                if content.strip():
                    texts.append(content.strip())
    
    with col2:
        st.header("📊 Analysis Results")
        
        if len(texts) >= 2:
            # Create text labels
            text_labels = [f"Text {i+1}" for i in range(len(texts))]
            
            # Analysis button
            if st.button("🔍 Analyze Similarity", type="primary"):
                results = {}
                
                # Process each selected model
                for model_name in selected_models:
                    with st.spinner(f"Processing with {model_name}..."):
                        embeddings = detector.generate_embeddings(texts, model_name)
                        if embeddings is not None:
                            similarity_matrix = detector.calculate_similarity_matrix(embeddings)
                            clones = detector.detect_clones(similarity_matrix, texts)
                            
                            results[model_name] = {
                                'similarity_matrix': similarity_matrix,
                                'clones': clones,
                                'avg_similarity': np.mean(similarity_matrix[np.triu_indices(len(texts), k=1)])
                            }
                
                # Display results
                if results:
                    st.success(f"✅ Analysis complete! Processed {len(texts)} texts with {len(selected_models)} model(s).")
                    
                    # Model comparison tabs
                    if len(selected_models) > 1:
                        tabs = st.tabs(selected_models)
                        
                        for idx, (model_name, tab) in enumerate(zip(selected_models, tabs)):
                            with tab:
                                display_model_results(results[model_name], texts, text_labels, detector, model_name)
                    else:
                        model_name = selected_models[0]
                        display_model_results(results[model_name], texts, text_labels, detector, model_name)
                    
                    # Model comparison summary
                    if len(selected_models) > 1:
                        st.subheader("🔬 Model Comparison Summary")
                        
                        comparison_data = []
                        for model_name in selected_models:
                            if model_name in results:
                                comparison_data.append({
                                    'Model': model_name,
                                    'Avg Similarity': f"{results[model_name]['avg_similarity']:.3f}",
                                    'Clones Detected': len(results[model_name]['clones']),
                                    'Max Similarity': f"{np.max(results[model_name]['similarity_matrix']):.3f}"
                                })
                        
                        if comparison_data:
                            df_comparison = pd.DataFrame(comparison_data)
                            st.dataframe(df_comparison, use_container_width=True)
        else:
            st.info("👆 Please enter at least 2 texts to perform similarity analysis.")
    
    # Documentation section
    with st.expander("📚 How It Works - Understanding Semantic Similarity"):
        st.markdown("""
        ### 🧠 Semantic Similarity for Plagiarism Detection
        
        **What are Embeddings?**
        - Embeddings convert text into numerical vectors that capture semantic meaning
        - Similar texts have similar vector representations
        - Unlike keyword matching, embeddings understand context and meaning
        
        **How We Detect Plagiarism:**
        1. **Preprocessing**: Clean and normalize input texts
        2. **Embedding Generation**: Convert texts to high-dimensional vectors
        3. **Similarity Calculation**: Use cosine similarity to measure vector similarity
        4. **Clone Detection**: Flag text pairs above the similarity threshold
        
        **Model Comparison:**
        - **all-MiniLM-L6-v2**: Lightweight, fast, good for general use
        - **all-mpnet-base-v2**: Higher quality embeddings, better accuracy
        - **paraphrase-multilingual**: Specialized for paraphrase detection
        
        **Similarity Score Interpretation:**
        - **0.9-1.0**: Highly likely plagiarism or near-duplicate
        - **0.8-0.9**: Potential plagiarism, needs review
        - **0.7-0.8**: Similar topics or themes
        - **<0.7**: Likely original content
        """)

def display_model_results(result, texts, text_labels, detector, model_name):
    """Display results for a specific model"""
    st.subheader(f"📊 Results for {model_name}")
    
    # Similarity matrix heatmap
    fig = detector.create_similarity_heatmap(result['similarity_matrix'], text_labels)
    st.plotly_chart(fig, use_container_width=True)
    
    # Clone detection results
    clones = result['clones']
    if clones:
        st.subheader("🚨 Potential Clones Detected")
        
        for clone in clones:
            similarity_pct = clone['similarity'] * 100
            
            # Color code based on similarity
            if similarity_pct >= 95:
                alert_type = "error"
                icon = "🔴"
            elif similarity_pct >= 85:
                alert_type = "warning"
                icon = "🟡"
            else:
                alert_type = "info"
                icon = "🟢"
            
            with st.container():
                st.markdown(f"### {icon} Similarity: {similarity_pct:.1f}%")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Text {clone['text1_idx'] + 1}:**")
                    st.text_area("", value=clone['text1_preview'], height=100, disabled=True, key=f"clone1_{model_name}_{clone['text1_idx']}_{clone['text2_idx']}")
                
                with col2:
                    st.markdown(f"**Text {clone['text2_idx'] + 1}:**")
                    st.text_area("", value=clone['text2_preview'], height=100, disabled=True, key=f"clone2_{model_name}_{clone['text1_idx']}_{clone['text2_idx']}")
                
                st.markdown("---")
    else:
        st.success("✅ No potential clones detected above the similarity threshold.")
    
    # Similarity matrix as table
    with st.expander("📋 Detailed Similarity Matrix"):
        df_similarity = pd.DataFrame(
            result['similarity_matrix'],
            index=text_labels,
            columns=text_labels
        )
        st.dataframe(df_similarity.round(3), use_container_width=True)

if __name__ == "__main__":
    main()