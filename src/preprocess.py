import pandas as pd
import numpy as np
import re
from collections import Counter, defaultdict
import pickle
import os
import json
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
import string
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
nltk_downloads = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 'omw-1.4']
for download in nltk_downloads:
    try:
        nltk.data.find(f'tokenizers/{download}' if download == 'punkt' else
                      f'corpora/{download}' if download in ['stopwords', 'wordnet', 'omw-1.4'] else
                      f'taggers/{download}')
    except LookupError:
        nltk.download(download)
        
 
def clip_token_ids(sequences, vocab_size):
    """
    Replace all token IDs >= vocab_size with 1 (<UNK>) to avoid embedding errors.
    """
    return [[token if token < vocab_size else 1 for token in seq] for seq in sequences]            

class LSTMQAPreprocessor:
    def __init__(self, csv_file_path=None, df=None):
        """
        Initialize the Grade A LSTM QA preprocessor with enhanced features
        """
        self.csv_file_path = csv_file_path
        self.df = df
        self.tokenizer_questions = None
        self.tokenizer_answers = None
        self.tokenizer_remarks = None
        self.tokenizer_improvements = None
        self.max_question_length = None
        self.max_answer_length = None
        self.max_remarks_length = None
        self.max_improvements_length = None

        self.vocab_size_questions = None
        self.vocab_size_answers = None
        self.vocab_size_remarks = None
        self.vocab_size_improvements = None
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Technical terms to preserve during preprocessing
        self.technical_preserve = {
            # Networking
            'tcp', 'udp', 'ip', 'dns', 'http', 'https', 'smtp', 'ftp', 'ssl', 'tls',
            'lan', 'wan', 'vpn', 'nat', 'dhcp', 'ospf', 'rip', 'bgp', 'vlan',
            'qos', 'snmp', 'ieee', 'osi', 'mac', 'arp', 'icmp', 'ppp', 'ethernet',
            # Programming & Web
            'api', 'url', 'uri', 'xml', 'html', 'css', 'js', 'sql', 'json', 'rest',
            'ajax', 'dom', 'mvc', 'orm', 'crud', 'jwt', 'oauth', 'cors', 'graphql',
            'nodejs', 'reactjs', 'angular', 'vue', 'python', 'java', 'javascript',
            # Data & AI/ML
            'rgb', 'yuv', 'jpeg', 'png', 'dct', 'dwt', 'lsb', 'msb',
            'lstm', 'gru', 'cnn', 'rnn', 'bert', 'gpt', 'transformer',
            'numpy', 'pandas', 'sklearn', 'tensorflow', 'pytorch', 'keras',
            'cuda', 'gpu', 'cpu', 'ram', 'ssd', 'hdd', 'io', 'os',
            # Cloud & DevOps
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'ci', 'cd',
            'git', 'github', 'gitlab', 'jenkins', 'ansible', 'terraform',
            # Security
            'firewall', 'intrusion', 'ids', 'ips', 'hashing', 'salting', 'md5', 'sha1', 'sha256',
            'rsa', 'aes', 'des', 'blowfish', 'twofish', 'cipher', 'encryption', 'decryption',
            'tls', 'ssl', 'certificate', 'publickey', 'privatekey', 'xss', 'csrf', 'sqli',
            'zero-day', 'cve', 'mitm', 'bruteforce', 'rce', 'dos', 'ddos',
            # Data Privacy
            'gdpr', 'ccpa', 'hipaa', 'anonymization', 'pseudonymization',
            'tokenization', 'k-anonymity', 'differentialprivacy', 'privacybudget',
            'consent', 'dataminimization', 'privacybydesign',
            # Malware Analysis
            'virustotal', 'anyrun', 'hybridanalysis', 'ida', 'ghidra', 'peid', 'peview',
            'floss', 'remnux', 'sandbox', 'signature', 'behavioral', 'static', 'dynamic',
            'obfuscation', 'packer', 'payload', 'dropper', 'keylogger', 'trojan',
            'ransomware', 'rootkit', 'botnet', 'spyware', 'adware',
            # Digital Forensics
            'chainofcustody', 'writeblocker', 'bitstream', 'evidence', 'hashmatch',
            'timelineanalysis', 'metadata', 'artifact', 'volatility', 'autopsy',
            'sleuthkit', 'hexeditor', 'memorydump', 'diskimage', 'forensiccopy',
            # Digital Watermarking
            'dct', 'dwt', 'watermarking', 'robustness', 'perceptibility', 'capacity',
            'invisiblewatermark', 'visiblewatermark', 'fragilewatermark',
            # Steganography
            'steganalysis', 'lsb', 'steganography', 'coverimage', 'payload', 'carrier',
            'spatialdomain', 'frequencydomain', 'invisibleink', 'stegdetect'
        }
        
        self.quality_keywords = {
            'high_quality': ['comprehensive', 'detailed', 'thorough', 'complete', 'accurate', 'precise'],
            'medium_quality': ['good', 'adequate', 'sufficient', 'reasonable', 'acceptable'],
            'improvement_needed': ['incomplete', 'unclear', 'vague', 'missing', 'incorrect', 'confusing']
        }
        print("Quality keywords initialized:")
    

    def load_data(self):
        """
        Load and validate CSV data with enhanced column detection
        """
        print("Loading data with enhanced validation...")
        try:
            if self.df is not None:
                print("Using provided DataFrame")
            elif self.csv_file_path:
                self.df = pd.read_csv(self.csv_file_path)
                print(f"Data loaded from {self.csv_file_path}")
            else:
                raise ValueError("Either csv_file_path or df must be provided")
            
            print(f"Shape: {self.df.shape}")
            print(f"Columns: {list(self.df.columns)}")
            
            # Enhanced column mapping
            original_columns = list(self.df.columns)
            column_mapping = {}
            
            for i, col in enumerate(original_columns):
                col_lower = col.lower().strip()
                col_original = col.strip()
                
                # Check for question, answer, remarks, improvements columns
                if (any(keyword in col_lower for keyword in ['quest', 'q']) or col_lower == 'question'):
                    column_mapping[col_original] = 'question'
                elif (any(keyword in col_lower for keyword in ['ans', 'a', 'solution', 'response']) or col_lower == 'answer'):
                    column_mapping[col_original] = 'answer'
                elif any(keyword in col_lower for keyword in ['remark', 'comment', 'note', 'feedback']):
                    column_mapping[col_original] = 'remarks'
                elif any(keyword in col_lower for keyword in ['improve', 'suggest', 'enhance', 'better']):
                    column_mapping[col_original] = 'improvements'
            
            # Fallback mapping for positional columns
            if not any(target == 'question' for target in column_mapping.values()) and len(original_columns) >= 1:
                column_mapping[original_columns[0]] = 'question'
            if not any(target == 'answer' for target in column_mapping.values()) and len(original_columns) >= 2:
                column_mapping[original_columns[1]] = 'answer'
            if not any(target == 'remarks' for target in column_mapping.values()) and len(original_columns) >= 3:
                column_mapping[original_columns[2]] = 'remarks'
            if not any(target == 'improvements' for target in column_mapping.values()) and len(original_columns) >= 4:
                column_mapping[original_columns[3]] = 'improvements'
            
            # Rename columns
            self.df = self.df.rename(columns=column_mapping)
            
            # Check required columns
            required_columns = ['question', 'answer']
            for col in required_columns:
                if col not in self.df.columns:
                    raise ValueError(f"Required column '{col}' not found in data")
            
            # Add optional columns if missing
            optional_columns = ['remarks', 'improvements']
            for col in optional_columns:
                if col not in self.df.columns:
                    self.df[col] = ''
                    print(f"Added empty '{col}' column")
            
            print(f"\nColumn mapping: {column_mapping}")
            print(f"Final columns: {list(self.df.columns)}")
            
            # Data preview
            print("\nData preview:")
            for idx, row in self.df.head(2).iterrows():
                print(f"\n--- Sample {idx+1} ---")
                print(f"Question: {str(row['question'])[:100]}...")
                print(f"Answer: {str(row['answer'])[:100]}...")
                if pd.notna(row['remarks']) and str(row['remarks']).strip():
                    print(f"Remarks: {str(row['remarks'])[:100]}...")
                if pd.notna(row['improvements']) and str(row['improvements']).strip():
                    print(f"Improvements: {str(row['improvements'])[:100]}...")
            print("-" * 50)
            
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    def advanced_text_cleaning(self, text, text_type='general'):
        """
        Advanced text cleaning with context-aware processing
        """
        if pd.isna(text) or text == '':
            return ""
        
        text = str(text).strip()
        text = ' '.join(text.split())
        
        # Context-specific cleaning
        if text_type == 'question':
            text = self._preserve_question_structure(text)
        elif text_type == 'answer':
            text = self._preserve_technical_content(text)
        elif text_type in ['remarks', 'improvements']:
            text = self._preserve_evaluative_content(text)
        
        text = self._preserve_technical_terms(text)
        text = self._expand_contractions(text)
        
        # Clean URLs, emails
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '<URL>', text)
        text = re.sub(r'\S+@\S+', '<EMAIL>', text)
        
        text = self._normalize_punctuation(text)
        
        # Tokenization and POS tagging
        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)
        filtered_tokens = self._intelligent_token_filtering(tokens, pos_tags, text_type)
        
        # Lemmatization for non-technical terms
        lemmatized_tokens = []
        for token in filtered_tokens:
            if token.lower() not in self.technical_preserve and len(token) > 2:
                lemmatized_tokens.append(self.lemmatizer.lemmatize(token.lower()))
            else:
                lemmatized_tokens.append(token)
        
        text = ' '.join(lemmatized_tokens)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def _preserve_question_structure(self, text):
        """Preserve question structure and interrogative words"""
        question_words = ['what', 'where', 'when', 'why', 'how', 'which', 'who', 'whose', 'whom']
        text_lower = text.lower()
        for qw in question_words:
            if qw in text_lower:
                text = re.sub(rf'\b{qw}\b', qw.upper(), text, flags=re.IGNORECASE)
        return text

    def _preserve_technical_content(self, text):
        """Preserve technical content and code snippets"""
        code_pattern = r'`([^`]+)`'
        code_matches = re.findall(code_pattern, text)
        
        for i, match in enumerate(code_matches):
            text = text.replace(f'`{match}`', f'<CODE{i}>')
        
        if not hasattr(self, '_code_cache'):
            self._code_cache = {}
        
        for i, match in enumerate(code_matches):
            self._code_cache[f'<CODE{i}>'] = match
        
        return text

    def _preserve_evaluative_content(self, text):
        """Preserve evaluative and sentiment language"""
        evaluative_patterns = [
            r'\b(very|extremely|highly|quite|rather|fairly|somewhat)\s+\w+',
            r'\b(excellent|good|poor|bad|great|terrible|amazing|awful)\b',
            r'\b(should|could|would|might|must|need to|ought to)\b'
        ]
        
        for pattern in evaluative_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                text = re.sub(pattern, match.upper(), text, flags=re.IGNORECASE)
        
        return text

    def _preserve_technical_terms(self, text):
        """Enhanced technical term preservation"""
        technical_pattern = r'\b(' + '|'.join(self.technical_preserve) + r')\b'
        tech_matches = re.findall(technical_pattern, text, re.IGNORECASE)
        
        text = text.lower()
        for match in tech_matches:
            text = re.sub(r'\b' + match.lower() + r'\b', match.upper(), text)
        
        return text

    def _expand_contractions(self, text):
        """Enhanced contraction expansion"""
        contractions = {
            "won't": "will not", "can't": "cannot", "n't": " not",
            "'re": " are", "'ve": " have", "'ll": " will",
            "'d": " would", "'m": " am", "'s": " is",
            "let's": "let us", "that's": "that is", "what's": "what is",
            "it's": "it is", "there's": "there is", "here's": "here is",
            "don't": "do not", "doesn't": "does not", "didn't": "did not",
            "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
            "won't": "will not", "wouldn't": "would not", "shouldn't": "should not",
            "couldn't": "could not", "mustn't": "must not"
        }
        
        for contraction, expansion in contractions.items():
            text = re.sub(r'\b' + re.escape(contraction) + r'\b', expansion, text, flags=re.IGNORECASE)
        
        return text

    def _normalize_punctuation(self, text):
        """Advanced punctuation normalization"""
        text = re.sub(r'[.,!?;:]{2,}', '.', text)
        text = re.sub(r'[^\w\s.,!?;:()\[\]{}\-/\\@#$%^&*+=|<>]', ' ', text)
        text = re.sub(r'\s*\(\s*', ' (', text)
        text = re.sub(r'\s*\)\s*', ') ', text)
        return text

    def _intelligent_token_filtering(self, tokens, pos_tags, text_type):
        """Intelligent token filtering based on POS tags and context"""
        filtered_tokens = []
        
        for token, pos in pos_tags:
            # Always keep technical terms
            if token.lower() in self.technical_preserve:
                filtered_tokens.append(token)
                continue
            
            # Keep meaningful tokens based on POS tags
            if pos in ['NN', 'NNS', 'NNP', 'NNPS',  # Nouns
                      'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',  # Verbs
                      'JJ', 'JJR', 'JJS',  # Adjectives
                      'RB', 'RBR', 'RBS',  # Adverbs
                      'WP', 'WRB']:  # Wh-words
                filtered_tokens.append(token)
            
            # Context-specific filtering
            elif text_type == 'question' and pos in ['WDT', 'WP$']:  # Which, Whose, etc
                filtered_tokens.append(token)
            elif text_type in ['remarks', 'improvements'] and pos in ['MD']:  # Should, could, might
                filtered_tokens.append(token)
            elif token in ['.', '?', '!', ',', ';', ':', '(', ')', '[', ']', '{', '}']:
                filtered_tokens.append(token)
            elif token.isdigit() or token in ['<URL>', '<EMAIL>'] or token.startswith('<CODE'):
                filtered_tokens.append(token)
            elif len(token) <= 3 and token.lower() in ['i', 'a', 'is', 'am', 'it', 'we', 'he', 'me',
                                                       'us', 'do', 'go', 'no', 'up', 'to', 'of', 'or', 'at', 'in', 'on', 'by', 'for', 'and', 'but', 'not',
                                                       'can', 'may', 'has', 'had', 'was', 'are', 'the']:
                if text_type != 'question' or token.lower() not in ['the', 'and', 'but']:
                    filtered_tokens.append(token)
        
        return filtered_tokens

    def assess_data_quality(self):
        """Assess and categorize data quality"""
        print("\nAssessing data quality...")
        quality_scores = []
        
        for idx, row in self.df.iterrows():
            score = 0
            details = {}
            
            # Question length assessment
            q_len = len(str(row['question']).split())
            if q_len >= 5:
                score += 2
                details['question_length'] = 'good'
            elif q_len >= 3:
                score += 1
                details['question_length'] = 'adequate'
            else:
                details['question_length'] = 'short'
            
            # Answer length assessment
            a_len = len(str(row['answer']).split())
            if a_len >= 15:
                score += 3
                details['answer_length'] = 'comprehensive'
            elif a_len >= 8:
                score += 2
                details['answer_length'] = 'good'
            elif a_len >= 5:
                score += 1
                details['answer_length'] = 'adequate'
            else:
                details['answer_length'] = 'short'
            
            # Technical content assessment
            tech_terms = sum(1 for term in self.technical_preserve
                           if term in str(row['question']).lower() or term in str(row['answer']).lower())
            if tech_terms >= 3:
                score += 2
                details['technical_content'] = 'high'
            elif tech_terms >= 1:
                score += 1
                details['technical_content'] = 'medium'
            else:
                details['technical_content'] = 'low'
            
            # Remarks and improvements assessment
            if pd.notna(row['remarks']) and len(str(row['remarks']).strip()) > 10:
                score += 1
                details['has_remarks'] = True
            
            if pd.notna(row['improvements']) and len(str(row['improvements']).strip()) > 10:
                score += 1
                details['has_improvements'] = True
            
            quality_scores.append({
                'index': idx,
                'score': score,
                'category': 'high' if score >= 7 else 'medium' if score >= 4 else 'low',
                'details': details
            })
        
        self.quality_assessment = quality_scores
        
        # Print quality distribution
        categories = [item['category'] for item in quality_scores]
        category_counts = Counter(categories)
        print(f"Quality distribution:")
        print(f" High quality: {category_counts['high']} ({category_counts['high']/len(self.df)*100:.1f}%)")
        print(f" Medium quality: {category_counts['medium']} ({category_counts['medium']/len(self.df)*100:.1f}%)")
        print(f" Low quality: {category_counts['low']} ({category_counts['low']/len(self.df)*100:.1f}%)")
        
        return quality_scores

    def clean_dataset(self):
        """Enhanced dataset cleaning with quality assessment"""
        print("\nCleaning Q&A dataset with quality assessment...")
        initial_len = len(self.df)
        
        print(f"Initial dataset size: {initial_len}")
        print(f"Missing questions: {self.df['question'].isna().sum()}")
        print(f"Missing answers: {self.df['answer'].isna().sum()}")
        print(f"Missing remarks: {self.df['remarks'].isna().sum()}")
        print(f"Missing improvements: {self.df['improvements'].isna().sum()}")
        
        # Remove rows with missing Q&A
        self.df = self.df.dropna(subset=['question', 'answer'])
        print(f"After removing missing Q&A: {len(self.df)} rows")
        
        # Fill missing optional columns
        self.df['remarks'] = self.df['remarks'].fillna('')
        self.df['improvements'] = self.df['improvements'].fillna('')
        
        # Clean all text columns
        print("Cleaning questions...")
        self.df['question_clean'] = self.df['question'].apply(
            lambda x: self.advanced_text_cleaning(x, 'question')
        )
        
        print("Cleaning answers...")
        self.df['answer_clean'] = self.df['answer'].apply(
            lambda x: self.advanced_text_cleaning(x, 'answer')
        )
        
        print("Cleaning remarks...")
        self.df['remarks_clean'] = self.df['remarks'].apply(
            lambda x: self.advanced_text_cleaning(x, 'remarks')
        )
        
        print("Cleaning improvements...")
        self.df['improvements_clean'] = self.df['improvements'].apply(
            lambda x: self.advanced_text_cleaning(x, 'improvements')
        )
        
        # Remove empty entries
        self.df = self.df[
            (self.df['question_clean'].str.len() > 0) &
            (self.df['answer_clean'].str.len() > 0)
        ]
        print(f"After removing empty entries: {len(self.df)} rows")
        
        # Quality filtering
        self.df = self.df[
            (self.df['question_clean'].str.len() >= 10) &
            (self.df['answer_clean'].str.len() >= 15) &
            (self.df['question_clean'].str.split().str.len() >= 3) &
            (self.df['answer_clean'].str.split().str.len() >= 4)
        ]
        print(f"After quality filtering: {len(self.df)} rows")
        
        # Remove near-duplicates
        print("Removing near-duplicates...")
        self.df = self._remove_near_duplicates()
        print(f"After duplicate removal: {len(self.df)} rows")
        
        # Assess quality
        self.assess_data_quality()
        
        # Reset index
        self.df = self.df.reset_index(drop=True)
        
        print(f"\nFinal cleaned dataset: {len(self.df)} Q&A pairs")
        
        # Sample cleaned data
        print("\nSample cleaned data:")
        for i in range(min(2, len(self.df))):
            print(f"\n--- Sample {i+1} ---")
            print(f"Q: {self.df.iloc[i]['question_clean']}")
            print(f"A: {self.df.iloc[i]['answer_clean'][:100]}...")
            if self.df.iloc[i]['remarks_clean']:
                print(f"R: {self.df.iloc[i]['remarks_clean'][:100]}...")
            if self.df.iloc[i]['improvements_clean']:
                print(f"I: {self.df.iloc[i]['improvements_clean'][:100]}...")
        
        return self.df

    def _remove_near_duplicates(self, similarity_threshold=0.85):
        """Remove near-duplicate questions using TF-IDF similarity"""
        if len(self.df) < 2:
            return self.df
        
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
        try:
            tfidf_matrix = vectorizer.fit_transform(self.df['question_clean'])
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            to_remove = set()
            for i in range(len(similarity_matrix)):
                if i in to_remove:
                    continue
                for j in range(i + 1, len(similarity_matrix)):
                    if similarity_matrix[i][j] > similarity_threshold:
                        # Keep higher quality sample if quality assessment exists
                        if hasattr(self, 'quality_assessment'):
                            score_i = next((item['score'] for item in self.quality_assessment if item['index'] == i), 0)
                            score_j = next((item['score'] for item in self.quality_assessment if item['index'] == j), 0)
                            if score_i >= score_j:
                                to_remove.add(j)
                            else:
                                to_remove.add(i)
                        else:
                            to_remove.add(j)
            
            self.df = self.df.drop(list(to_remove)).reset_index(drop=True)
            print(f"Removed {len(to_remove)} near-duplicate questions")
            print(f"Remaining questions: {len(self.df)}")
            
        except Exception as e:
            print(f"Could not remove duplicates: {e}")
        
        return self.df

    def enhanced_text_statistics(self):
        """Enhanced text analysis with multi-column support"""
        print("\nAnalyzing enhanced text statistics...")
        
        columns_to_analyze = {
            'question_clean': 'questions',
            'answer_clean': 'answers',
            'remarks_clean': 'remarks',
            'improvements_clean': 'improvements'
        }
        
        statistics = {}
        
        for col, name in columns_to_analyze.items():
            if col in self.df.columns:
                texts = self.df[col].tolist()
                texts = [text for text in texts if text and len(text.strip()) > 0]
                
                if texts:
                    lengths = [len(text.split()) for text in texts]
                    char_lengths = [len(text) for text in texts]
                    
                    stats = {
                        'count': len(texts),
                        'word_lengths': {
                            'min': min(lengths),
                            'max': max(lengths),
                            'mean': np.mean(lengths),
                            'median': np.median(lengths),
                            'p75': np.percentile(lengths, 75),
                            'p90': np.percentile(lengths, 90),
                            'p95': np.percentile(lengths, 95)
                        },
                        'char_lengths': {
                            'min': min(char_lengths),
                            'max': max(char_lengths),
                            'mean': np.mean(char_lengths),
                            'median': np.median(char_lengths)
                        }
                    }
                    
                    statistics[name] = stats
                    print(f"\n{name.title()} statistics:")
                    print(f" Count: {stats['count']}")
                    print(f" Word length - Min: {stats['word_lengths']['min']}, Max: {stats['word_lengths']['max']}, Mean: {stats['word_lengths']['mean']:.1f}, P90: {stats['word_lengths']['p90']:.1f}")
        
        # Set optimal sequence lengths
        self.max_question_length = 100
        self.max_answer_length = 100
        
        if 'remarks' in statistics and statistics['remarks']['count'] > 0:
            self.max_remarks_length = min(max(15, int(statistics['remarks']['word_lengths']['p90'] * 1.3)), 100)
        else:
            self.max_remarks_length = 50
        
        if 'improvements' in statistics and statistics['improvements']['count'] > 0:
            self.max_improvements_length = min(max(15, int(statistics['improvements']['word_lengths']['p90'] * 1.3)), 100)
        else:
            self.max_improvements_length = 50
        
        print(f"\nOptimal sequence lengths:")
        print(f" Questions: {self.max_question_length}")
        print(f" Answers: {self.max_answer_length}")
        print(f" Remarks: {self.max_remarks_length}")
        print(f" Improvements: {self.max_improvements_length}")
        
        self.text_statistics = statistics
        return statistics

    def create_enhanced_vocabulary(self):
        """
        Create enhanced vocabulary with proper tokenizer configuration
        """
        print("\nCreating enhanced vocabulary...")
        
        # Determine optimal vocabulary sizes
        vocab_configs = {
            'questions': {
                'texts': self.df['question_clean'].tolist(),
                'base_vocab_size': 5000,
                'min_freq': 2
            },
            'answers': {
                'texts': self.df['answer_clean'].tolist(),
                'base_vocab_size': 8000,
                'min_freq': 2
            },
            'remarks': {
                'texts': [text for text in self.df['remarks_clean'].tolist() if text.strip()],
                'base_vocab_size': 3000,
                'min_freq': 1
            },
            'improvements': {
                'texts': [text for text in self.df['improvements_clean'].tolist() if text.strip()],
                'base_vocab_size': 3000,
                'min_freq': 1
            }
        }
        
        # Create tokenizers
        self.tokenizer_questions = self._create_tokenizer(
            vocab_configs['questions']['texts'],
            vocab_configs['questions']['base_vocab_size'],
            vocab_configs['questions']['min_freq']
        )
        self.vocab_size_questions = len(self.tokenizer_questions.word_index) + 1
        
        self.tokenizer_answers = self._create_tokenizer(
            vocab_configs['answers']['texts'],
            vocab_configs['answers']['base_vocab_size'],
            vocab_configs['answers']['min_freq']
        )
        self.vocab_size_answers = len(self.tokenizer_answers.word_index) + 1
        
        self.tokenizer_remarks = self._create_tokenizer(
            vocab_configs['remarks']['texts'],
            vocab_configs['remarks']['base_vocab_size'],
            vocab_configs['remarks']['min_freq']
        )
        self.vocab_size_remarks = len(self.tokenizer_remarks.word_index) + 1
        
        self.tokenizer_improvements = self._create_tokenizer(
            vocab_configs['improvements']['texts'],
            vocab_configs['improvements']['base_vocab_size'],
            vocab_configs['improvements']['min_freq']
        )
        self.vocab_size_improvements = len(self.tokenizer_improvements.word_index) + 1
        
        print(f"Vocabulary sizes:")
        print(f" Questions: {self.vocab_size_questions}")
        print(f" Answers: {self.vocab_size_answers}")
        print(f" Remarks: {self.vocab_size_remarks}")
        print(f" Improvements: {self.vocab_size_improvements}")
        
        return {
            'questions': self.vocab_size_questions,
            'answers': self.vocab_size_answers,
            'remarks': self.vocab_size_remarks,
            'improvements': self.vocab_size_improvements
        }

    
    
    def _create_tokenizer(self, texts, vocab_size, min_freq=1):
        """
        Create a tokenizer with a controlled vocabulary and <UNK> support.
        """
        if not texts or len(texts) == 0:
            tokenizer = Tokenizer(
                num_words=1000,
                oov_token='<UNK>',
                filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
            )
            tokenizer.fit_on_texts(['empty'])
            return tokenizer

        # Filter out empty strings
        texts = [text for text in texts if text.strip()]

        tokenizer = Tokenizer(
            num_words=vocab_size,
            oov_token='<UNK>',
            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
        )
        tokenizer.fit_on_texts(texts)

        # Apply min frequency filter if needed
        if min_freq > 1:
            word_freq = tokenizer.word_counts
            filtered_words = {word: count for word, count in word_freq.items() if count >= min_freq}

            tokenizer = Tokenizer(
                num_words=min(vocab_size, len(filtered_words) + 1),
                oov_token='<UNK>',
                filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
            )
            tokenizer.fit_on_texts(texts)

        return tokenizer

    def prepare_sequences(self):
        """
        Prepare sequences with CRITICAL token ID filtering using clip_token_ids()
        """
        print("\nPreparing sequences with token ID filtering...")

        # Prepare fallback texts
        remarks_texts = [text for text in self.df['remarks_clean'].tolist() if text.strip()]
        if not remarks_texts:
            remarks_texts = ['']

        improvements_texts = [text for text in self.df['improvements_clean'].tolist() if text.strip()]
        if not improvements_texts:
            improvements_texts = ['']

        # Tokenize and clip
        print("Generating and filtering question sequences...")
        question_sequences = self.tokenizer_questions.texts_to_sequences(self.df['question_clean'])
        question_sequences = clip_token_ids(question_sequences, self.vocab_size_questions)

        print("Generating and filtering answer sequences...")
        answer_sequences = self.tokenizer_answers.texts_to_sequences(self.df['answer_clean'])
        answer_sequences = clip_token_ids(answer_sequences, self.vocab_size_answers)

        print("Generating and filtering remarks sequences...")
        remarks_sequences = self.tokenizer_remarks.texts_to_sequences(remarks_texts)
        remarks_sequences = clip_token_ids(remarks_sequences, self.vocab_size_remarks)

        print("Generating and filtering improvements sequences...")
        improvements_sequences = self.tokenizer_improvements.texts_to_sequences(improvements_texts)
        improvements_sequences = clip_token_ids(improvements_sequences, self.vocab_size_improvements)

        # Pad sequences
        print("Padding sequences...")
        question_sequences = pad_sequences(
            question_sequences, maxlen=self.max_question_length, padding='post', truncating='post'
        )
        answer_sequences = pad_sequences(
            answer_sequences, maxlen=self.max_answer_length, padding='post', truncating='post'
        )
        remarks_sequences = pad_sequences(
            remarks_sequences, maxlen=self.max_remarks_length, padding='post', truncating='post'
        )
        improvements_sequences = pad_sequences(
            improvements_sequences, maxlen=self.max_improvements_length, padding='post', truncating='post'
        )

        # Safety check
        self._verify_token_ranges(question_sequences, self.vocab_size_questions, "questions")
        self._verify_token_ranges(answer_sequences, self.vocab_size_answers, "answers")
        self._verify_token_ranges(remarks_sequences, self.vocab_size_remarks, "remarks")
        self._verify_token_ranges(improvements_sequences, self.vocab_size_improvements, "improvements")

        # Final output
        print(f"Sequence shapes:")
        print(f" Questions: {question_sequences.shape}")
        print(f" Answers: {answer_sequences.shape}")
        print(f" Remarks: {remarks_sequences.shape}")
        print(f" Improvements: {improvements_sequences.shape}")

        self.sequences = {
            'questions': question_sequences,
            'answers': answer_sequences,
            'remarks': remarks_sequences,
            'improvements': improvements_sequences
        }

        return self.sequences


    def _verify_token_ranges(self, sequences, vocab_size, sequence_type):
        """
        Verify that all token IDs are within valid range
        """
        max_token = np.max(sequences)
        min_token = np.min(sequences)
        
        print(f"{sequence_type.title()} token range: [{min_token}, {max_token}] (vocab_size: {vocab_size})")
        
        if max_token >= vocab_size:
            print(f"⚠️  WARNING: {sequence_type} has tokens >= vocab_size ({vocab_size})")
            # Count out-of-bounds tokens
            out_of_bounds = np.sum(sequences >= vocab_size)
            print(f"   Out-of-bounds tokens: {out_of_bounds}")
        else:
            print(f"✅ {sequence_type.title()} token IDs are within valid range")

    def create_quality_labels(self):
        """
        Create quality labels for multi-task learning
        """
        print("\nCreating quality labels...")
        
        if not hasattr(self, 'quality_assessment'):
            self.assess_data_quality()
        
        # Create categorical quality labels
        quality_categories = []
        for assessment in self.quality_assessment:
            category = assessment['category']
            if category == 'high':
                quality_categories.append([1, 0, 0])  # high, medium, low
            elif category == 'medium':
                quality_categories.append([0, 1, 0])
            else:
                quality_categories.append([0, 0, 1])
        
        quality_labels = np.array(quality_categories)
        
        # Create binary labels for specific quality aspects
        has_remarks = []
        has_improvements = []
        
        for _, row in self.df.iterrows():
            has_remarks.append(1 if row['remarks_clean'].strip() else 0)
            has_improvements.append(1 if row['improvements_clean'].strip() else 0)
        
        self.quality_labels = {
            'overall_quality': quality_labels,
            'has_remarks': np.array(has_remarks),
            'has_improvements': np.array(has_improvements)
        }
        
        print(f"Quality label shapes:")
        print(f" Overall quality: {self.quality_labels['overall_quality'].shape}")
        print(f" Has remarks: {self.quality_labels['has_remarks'].shape}")
        print(f" Has improvements: {self.quality_labels['has_improvements'].shape}")
        
        return self.quality_labels

    # def create_train_test_split(self, test_size=0.2, random_state=42):
    #     """
    #     Create stratified train-test split
    #     """
    #     print(f"\nCreating train-test split (test_size={test_size})...")
        
    #     if not hasattr(self, 'sequences'):
    #         raise ValueError("Sequences not prepared. Call prepare_sequences() first.")
        
    #     if not hasattr(self, 'quality_labels'):
    #         self.create_quality_labels()
        
    #     # Use stratified split based on overall quality
    #     quality_cats = np.argmax(self.quality_labels['overall_quality'], axis=1)
        
    #     # Create indices for splitting
    #     indices = np.arange(len(self.df))
        
    #     train_idx, test_idx = train_test_split(
    #         indices,
    #         test_size=test_size,
    #         random_state=random_state,
    #         stratify=quality_cats
    #     )
        
    #     # Split sequences
    #     train_data = {}
    #     test_data = {}
        
    #     for key, sequences in self.sequences.items():
    #         train_data[key] = sequences[train_idx]
    #         test_data[key] = sequences[test_idx]
        
    #     # Split labels
    #     train_labels = {}
    #     test_labels = {}
        
    #     for key, labels in self.quality_labels.items():
    #         train_labels[key] = labels[train_idx]
    #         test_labels[key] = labels[test_idx]
        
    #     self.train_data = train_data
    #     self.test_data = test_data
    #     self.train_labels = train_labels
    #     self.test_labels = test_labels
        
    #     print(f"Train set size: {len(train_idx)}")
    #     print(f"Test set size: {len(test_idx)}")
        
    #     # Print quality distribution
    #     train_quality_dist = np.bincount(np.argmax(train_labels['overall_quality'], axis=1))
    #     test_quality_dist = np.bincount(np.argmax(test_labels['overall_quality'], axis=1))
        
    #     print(f"Train quality distribution: High={train_quality_dist[0]}, Medium={train_quality_dist[1]}, Low={train_quality_dist[2]}")
    #     print(f"Test quality distribution: High={test_quality_dist[0]}, Medium={test_quality_dist[1]}, Low={test_quality_dist[2]}")
        
    #     return {
    #         'train_data': train_data,
    #         'test_data': test_data,
    #         'train_labels': train_labels,
    #         'test_labels': test_labels,
    #         'train_idx': train_idx,
    #         'test_idx': test_idx
    #     }

    def create_train_test_split(self, test_size=0.2, random_state=42):
        """
        Create a simple train-test split without stratification.
        Safely handles partial or dummy sequences.
        """
        print(f"\nCreating train-test split (test_size={test_size}) without stratification...")

        if not hasattr(self, 'sequences'):
            raise ValueError("Sequences not prepared. Call prepare_sequences() first.")

        if not hasattr(self, 'quality_labels'):
            self.create_quality_labels()

        total_samples = len(self.df)
        indices = np.arange(total_samples)

        # Simple shuffled split
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            shuffle=True
        )

        # Filter sequences safely
        train_data, test_data = {}, {}
        for key, value in self.sequences.items():
            arr = np.array(value)
            if arr.shape[0] == total_samples:
                train_data[key] = arr[train_idx]
                test_data[key] = arr[test_idx]
            else:
                # Leave mismatched or dummy arrays untouched
                train_data[key] = arr
                test_data[key] = arr

        # Filter labels safely
        train_labels = {
            key: np.array(value)[train_idx]
            for key, value in self.quality_labels.items()
        }
        test_labels = {
            key: np.array(value)[test_idx]
            for key, value in self.quality_labels.items()
        }

        # Save results
        self.train_data = train_data
        self.test_data = test_data
        self.train_labels = train_labels
        self.test_labels = test_labels

        print(f"✅ Train set size: {len(train_idx)}")
        print(f"✅ Test set size: {len(test_idx)}")

        # Print quality label distribution (optional)
        train_quality_dist = np.bincount(np.argmax(train_labels['overall_quality'], axis=1), minlength=3)
        test_quality_dist = np.bincount(np.argmax(test_labels['overall_quality'], axis=1), minlength=3)

        print(f"📊 Train quality distribution: High={train_quality_dist[0]}, Medium={train_quality_dist[1]}, Low={train_quality_dist[2]}")
        print(f"📊 Test quality distribution: High={test_quality_dist[0]}, Medium={test_quality_dist[1]}, Low={test_quality_dist[2]}")

        return {
            'train_data': train_data,
            'test_data': test_data,
            'train_labels': train_labels,
            'test_labels': test_labels,
            'train_idx': train_idx,
            'test_idx': test_idx
        }

    
    def save_preprocessed_data(self, save_dir='preprocessed_data'):
        """
        Save all preprocessed data, tokenizers, vocab sizes, sequences, and config.
        """
        print(f"\nSaving preprocessed data to {save_dir}...")

        os.makedirs(save_dir, exist_ok=True)

        # Save cleaned dataframe
        self.df.to_csv(os.path.join(save_dir, 'cleaned_qa_data.csv'), index=False)

        # Save tokenizers together and individually
        tokenizers = {
            'questions': self.tokenizer_questions,
            'answers': self.tokenizer_answers,
            'remarks': self.tokenizer_remarks,
            'improvements': self.tokenizer_improvements
        }
        with open(os.path.join(save_dir, 'tokenizers.pkl'), 'wb') as f:
            pickle.dump(tokenizers, f)

        # Also save tokenizers separately for loader compatibility
        for name, tok in tokenizers.items():
            with open(os.path.join(save_dir, f'tokenizer_{name}.pkl'), 'wb') as f:
                pickle.dump(tok, f)

        # Save vocab sizes
        vocab_sizes = {
            'questions': self.vocab_size_questions,
            'answers': self.vocab_size_answers,
            'remarks': self.vocab_size_remarks,
            'improvements': self.vocab_size_improvements
        }
        with open(os.path.join(save_dir, 'vocab_sizes.pkl'), 'wb') as f:
            pickle.dump(vocab_sizes, f)

        # Save sequences (ensure token IDs are valid)
        if hasattr(self, 'sequences'):
            for name, seqs in self.sequences.items():
                max_id = np.max(seqs)
                vocab_size = vocab_sizes.get(name)
                if vocab_size and max_id >= vocab_size:
                    print(f"⚠️  Warning: Max token ID in {name} = {max_id}, exceeds vocab_size={vocab_size}")
            np.savez_compressed(os.path.join(save_dir, 'sequences.npz'), **self.sequences)

        # Save train/test data + labels if available
        if hasattr(self, 'train_data'):
            np.savez_compressed(os.path.join(save_dir, 'train_data.npz'), **self.train_data)
            np.savez_compressed(os.path.join(save_dir, 'test_data.npz'), **self.test_data)
            np.savez_compressed(os.path.join(save_dir, 'train_labels.npz'), **self.train_labels)
            np.savez_compressed(os.path.join(save_dir, 'test_labels.npz'), **self.test_labels)

        # Save configuration and metadata
        config = {
            'vocab_sizes': vocab_sizes,
            'sequence_lengths': {
                'questions': self.max_question_length,
                'answers': self.max_answer_length,
                'remarks': self.max_remarks_length,
                'improvements': self.max_improvements_length
            },
            'dataset_info': {
                'total_samples': len(self.df),
                'columns': list(self.df.columns),
                'preprocessing_date': datetime.now().isoformat()
            }
        }

        if hasattr(self, 'text_statistics'):
            config['text_statistics'] = self.text_statistics
        if hasattr(self, 'quality_assessment'):
            config['quality_assessment'] = self.quality_assessment

        with open(os.path.join(save_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2, default=str)

        print(f"✅ All data saved to {save_dir}/")
        print(f"   - cleaned_qa_data.csv")
        print(f"   - tokenizers.pkl + tokenizer_*.pkl")
        print(f"   - vocab_sizes.pkl")
        print(f"   - sequences.npz")
        print(f"   - train/test splits")
        print(f"   - config.json")


    def load_preprocessed_data(self, save_dir='preprocessed_data'):
        """
        Load previously saved preprocessed data
        """
        print(f"Loading preprocessed data from {save_dir}...")
        
        # Load cleaned dataframe
        self.df = pd.read_csv(os.path.join(save_dir, 'cleaned_qa_data.csv'))
        
        # Load tokenizers
        with open(os.path.join(save_dir, 'tokenizers.pkl'), 'rb') as f:
            tokenizers = pickle.load(f)
        
        self.tokenizer_questions = tokenizers['questions']
        self.tokenizer_answers = tokenizers['answers']
        self.tokenizer_remarks = tokenizers['remarks']
        self.tokenizer_improvements = tokenizers['improvements']
        
        # Load configuration
        with open(os.path.join(save_dir, 'config.json'), 'r') as f:
            config = json.load(f)
        
        vocab_sizes = config['vocab_sizes']
        self.vocab_size_questions = vocab_sizes['questions']
        self.vocab_size_answers = vocab_sizes['answers']
        self.vocab_size_remarks = vocab_sizes['remarks']
        self.vocab_size_improvements = vocab_sizes['improvements']
        
        sequence_lengths = config['sequence_lengths']
        self.max_question_length = sequence_lengths['questions']
        self.max_answer_length = sequence_lengths['answers']
        self.max_remarks_length = sequence_lengths['remarks']
        self.max_improvements_length = sequence_lengths['improvements']
        
        # Load sequences if available
        if os.path.exists(os.path.join(save_dir, 'sequences.npz')):
            sequences_data = np.load(os.path.join(save_dir, 'sequences.npz'))
            self.sequences = {key: sequences_data[key] for key in sequences_data.files}
        
        # Load train-test split if available
        if os.path.exists(os.path.join(save_dir, 'train_data.npz')):
            train_data = np.load(os.path.join(save_dir, 'train_data.npz'))
            test_data = np.load(os.path.join(save_dir, 'test_data.npz'))
            train_labels = np.load(os.path.join(save_dir, 'train_labels.npz'))
            test_labels = np.load(os.path.join(save_dir, 'test_labels.npz'))
            
            self.train_data = {key: train_data[key] for key in train_data.files}
            self.test_data = {key: test_data[key] for key in test_data.files}
            self.train_labels = {key: train_labels[key] for key in train_labels.files}
            self.test_labels = {key: test_labels[key] for key in test_labels.files}
        
        # Load additional data
        if 'text_statistics' in config:
            self.text_statistics = config['text_statistics']
        if 'quality_assessment' in config:
            self.quality_assessment = config['quality_assessment']
        
        print("✅ All preprocessed data loaded successfully!")
        return True

    def run_complete_preprocessing(self):
        """
        Run the complete preprocessing pipeline
        """
        print("🚀 Starting complete LSTM QA preprocessing pipeline...")
        print("=" * 60)
        
        # Step 1: Load data
        if not self.load_data():
            print("❌ Failed to load data")
            return False
        
        # Step 2: Clean dataset
        cleaned_df = self.clean_dataset()
        if len(cleaned_df) == 0:
            print("❌ No data remaining after cleaning")
            return False
        
        # Step 3: Analyze text statistics
        self.enhanced_text_statistics()
        
        # Step 4: Create vocabulary
        self.create_enhanced_vocabulary()
        
        # Step 5: Prepare sequences with token filtering
        self.prepare_sequences()
        
        # Step 6: Create quality labels
        self.create_quality_labels()
        
        # Step 7: Create train-test split
        self.create_train_test_split()
        
        # Step 8: Save all preprocessed data
        self.save_preprocessed_data()
        
        print("\n" + "=" * 60)
        print("✅ Complete preprocessing pipeline finished successfully!")
        print(f"📊 Final dataset: {len(self.df)} Q&A pairs")
        print(f"🔤 Vocabulary sizes: Q={self.vocab_size_questions}, A={self.vocab_size_answers}")
        print(f"📏 Sequence lengths: Q={self.max_question_length}, A={self.max_answer_length}")
        print("💾 All data saved for model training")
        
        return True

    def get_model_ready_data(self):
        """
        Get data ready for model training
        """
        if not hasattr(self, 'train_data') or not hasattr(self, 'test_data'):
            print("❌ Train-test split not available. Run preprocessing first.")
            return None
        
        model_data = {
            'train_data': self.train_data,
            'test_data': self.test_data,
            'train_labels': self.train_labels,
            'test_labels': self.test_labels,
            'vocab_sizes': {
                'questions': self.vocab_size_questions,
                'answers': self.vocab_size_answers,
                'remarks': self.vocab_size_remarks,
                'improvements': self.vocab_size_improvements
            },
            'sequence_lengths': {
                'questions': self.max_question_length,
                'answers': self.max_answer_length,
                'remarks': self.max_remarks_length,
                'improvements': self.max_improvements_length
            },
            'tokenizers': {
                'questions': self.tokenizer_questions,
                'answers': self.tokenizer_answers,
                'remarks': self.tokenizer_remarks,
                'improvements': self.tokenizer_improvements
            }
        }
        
        return model_data


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    print("🔧 LSTM QA Preprocessor - Example Usage")
    print("=" * 50)
    
    # Initialize preprocessor
    # Option 1: From CSV file
    preprocessor = LSTMQAPreprocessor(csv_file_path='dataset.csv')
    
    # Option 2: From DataFrame
    # import pandas as pd
    # df = pd.read_csv('your_qa_data.csv')
    # preprocessor = LSTMQAPreprocessor(df=df)
    
    # Run complete preprocessing
    success = preprocessor.run_complete_preprocessing()
    
    if success:
        # Get model-ready data
        model_data = preprocessor.get_model_ready_data()
        print("🎯 Data is ready for LSTM model training!")
    
    print("\n📝 Usage Notes:")
    print("1. Ensure your CSV has 'question' and 'answer' columns")
    print("2. Optional: 'remarks' and 'improvements' columns")
    print("3. The preprocessor handles missing data automatically")
    print("4. All token IDs are safely filtered to prevent out-of-bounds errors")
    print("5. Quality assessment is performed automatically")
    print("6. Data is saved for future use")

