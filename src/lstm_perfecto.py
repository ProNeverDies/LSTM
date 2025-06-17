import os
import json
import numpy as np
import pickle
import re
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.models import load_model as _lm
# Keras imports for building models
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Embedding, Dropout,
    LayerNormalization, Add, Concatenate, Bidirectional,
    MultiHeadAttention, GlobalAveragePooling1D, GRU,
    BatchNormalization, Conv1D, MaxPooling1D
)
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau,
    ModelCheckpoint, CSVLogger, TensorBoard,
    LearningRateScheduler
)
from tensorflow.keras.regularizers import l2
from groq import Groq
import warnings
warnings.filterwarnings("ignore")

class CybersecurityLLM:
    """
    Groq-LLM wrapper. This includes questions about cybersecurity.
    """
    def __init__(self, api_key=None, model_name="llama3-8b-8192"):
        os.environ["GROQ_API_KEY"]=os.getenv('GROQ_AOI_KEY',api_key)
        self.api_key = "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" or os.getenv('GROQ_API_KEY')
        if not self.api_key:
            raise ValueError("Groq API key is required. Set GROQ_API_KEY or pass it directly.")
        self.client = Groq(api_key=self.api_key)
        self.model_name = model_name
        self.max_tokens = 150
        self.system_prompt = """
You are a technical expert AI assistant specializing in:
- Cybersecurity fundamentals, best practices, and threat detection
- Data privacy & protection regulations
- Malware analysis and reverse engineering
- Computer networking and network security
- Operating system security & design
- Artificial intelligence security implications and AI concepts
- Digital forensics techniques and incident response
- Steganography & watermarking methods
- Cryptography (symmetric, asymmetric, hashing, PKI, etc.)
- Penetration testing and vulnerability management
- DevSecOps, cloud security and container security
- General technical questions on programming, OS, and computer science

Provide accurate, concise, educational responses. Keep answers under 150 words and format clearly for learning.
"""
        self._compile_keyword_pattern()

    def _compile_keyword_pattern(self):
        """Build a single case-insensitive regex that triggers the LLM whenever ANY of the following keywords appear."""
        keywords = [
            # ── Core Cybersecurity ─────────────────────────────────────────────────
            r"\bcybersecurity\b",           r"\bsecurity\b",
            r"\binformation security\b",     r"\binfosec\b",
            r"\bdata protection\b",         r"\bdata privacy\b",
            r"\bprivacy\b",                 r"\bcompliance\b",
            r"\brisk management\b",         r"\bdisaster recovery\b",

            # ── Malware & Forensics ───────────────────────────────────────────────
            r"\bmalware\b",          r"\bvirus\b",
            r"\btrojan\b",           r"\bransomware\b",
            r"\bspyware\b",          r"\brootkit\b",
            r"\bworm\b",             r"\badware\b",
            r"\bbotnet\b",           r"\breverse engineering\b",
            r"\bforensics\b",        r"\bdigital forensics\b",
            r"\bincident response\b",r"\bthreat hunting\b",

            # ── Steganography & Watermarking ───────────────────────────────────────
            r"\bsteganography\b",    r"\bwatermarking\b",
            r"\bdct\b",              r"\bdwt\b",
            r"\baudio steganography\b", r"\btransform-domain steganography\b",

            # ── Penetration Testing & Vulnerability ─────────────────────────────────
            r"\bpen(et|et)ration testing\b", r"\bpenetration testing\b",
            r"\bpentest\b",          r"\bvulnerability management\b",
            r"\bexploit\b",          r"\bpatching\b",
            r"\bbug bounty\b",       r"\bred teaming\b",
            r"\bblue team\b",        r"\bpurple team\b",

            # ── Networking & Infrastructure ────────────────────────────────────────
            r"\bnetwork security\b",  r"\bcomputer networks\b",
            r"\bfirewall\b",          r"\bids\b",          r"\bips\b",
            r"\bvpn\b",               r"\bproxy\b",
            r"\bcloud security\b",    r"\bcontainer security\b",
            r"\bkubernetes security\b", r"\bdevsecops\b",
            r"\binfrastructure as code\b", r"\bcloud misconfigurations\b",

            # ── Authentication & Access Control ────────────────────────────────────
            r"\bauthentication\b",   r"\bauthorization\b",
            r"\bmfa\b",              r"\b2fa\b",
            r"\bsso\b",              r"\boauth\b",          r"\bsaml\b",
            r"\bidentity management\b", r"\bleast privilege\b",

            # ── Cryptography & PKI ─────────────────────────────────────────────────
            r"\bencryption\b",       r"\bdecryption\b",
            r"\bhashing\b",          r"\bssl\b",            r"\btls\b",
            r"\bcert(ificate)?\b",   r"\bpk\b",             r"\brsa\b",
            r"\baes\b",              r"\bkey exchange\b",
            r"\bpublic key\b",       r"\bprivate key\b",
            r"\bdigital signature\b",r"\bhash-based\b",
            r"\belliptic curve\b",   r"\bgnupg\b",

            # ── Operating Systems & Virtualization ─────────────────────────────────
            r"\boperating system\b", r"\bwindows security\b",
            r"\blinux security\b",   r"\bros security\b",
            r"\bmemory forensics\b",

            # ── Artificial Intelligence & ML ───────────────────────────────────────
            r"\bartificial intelligence\b", r"\bai security\b",
            r"\bmachine learning\b",       r"\bdeep learning\b",
            r"\bneural networks\b",        r"\bml security\b",

            # ── Database & Application Security ────────────────────────────────────
            r"\bsql injection\b",    r"\bxss\b",             r"\bcsrf\b",
            r"\bclickjacking\b",     r"\bdirectory traversal\b",
            r"\bsession hijacking\b",r"\btoken manipulation\b",
            r"\binput validation\b", r"\brate limiting\b",

            # ── Standards, Regulations & Compliance ─────────────────────────────────
            r"\bgdpr\b",             r"\bhipaa\b",           r"\bpci dss\b",
            r"\biso 27001\b",        r"\bnist\b",            r"\bndpr\b",
            r"\bcisa\b",             r"\bcmmc\b",            r"\bsoc 2\b",
            r"\bccpa\b",             r"\bsoar\b",

            # ── Miscellaneous / General Tech ───────────────────────────────────────
            r"\bprogramming\b",      r"\bdata structures\b",
            r"\balgorithms\b",       r"\boperating systems\b",
            r"\bcompiler\b",         r"\bcomputer architecture\b",
            r"\bnumerical methods\b",r"\bcloud computing\b",
            r"\bdevops\b",           r"\bapi development\b",
            r"\bdatabase design\b",  r"\bsoftware engineering\b"
        ]

        pattern = r"(?i)(" + r"|".join(keywords) + r")"
        self._keyword_regex = re.compile(pattern)

    def is_cybersecurity_related(self, question: str) -> bool:
        """Return True if the question matches any of our expanded keywords."""
        if not question or not question.strip():
            return False
        return bool(self._keyword_regex.search(question))

    def generate_answer(self, question: str, context: str = None) -> str:
        """Generate an answer using Groq LLM."""
        try:
            prompt = f"Question: {question}"
            if context:
                prompt = f"Context: {context}\n\n{prompt}"
            
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                model=self.model_name,
                max_tokens=self.max_tokens,
                temperature=0.7,
                top_p=0.9,
                stream=False,
            )
            
            answer = chat_completion.choices[0].message.content.strip()
            return self._format_answer(answer)
        except Exception as e:
            print(f"[LLM ERROR] {e}")
            print("\n")
            return "Sorry, the LLM failed to generate an answer."

    def _format_answer(self, answer: str) -> str:
        """Simple formatter to indent bullet points, remove extra blank lines, and truncate at ~140 words."""
        lines = answer.split("\n")
        formatted = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.endswith(":"):
                formatted.append(line)
            elif line.startswith("-") or line.startswith("•"):
                formatted.append(f"  {line}")
            else:
                formatted.append(line)
        
        out = "\n".join(formatted)
        words = out.split()             
        if len(words) > 140:            
            out = " ".join(words[:140]) + "..."
        return out


class EnhancedLSTMQAPreprocessorLoader:
    """Enhanced preprocessor with better text processing capabilities."""
    
    def __init__(self, preprocessor_path: str):
        self.preprocessor_path = preprocessor_path
        self.has_remarks = False
        self.has_improvements = False
        self._load_preprocessor_data()

    def _load_preprocessor_data(self):
        print("Loading enhanced preprocessor data from:", self.preprocessor_path)
        
        # Load config
        cfg_file = os.path.join(self.preprocessor_path, "config.json")
        with open(cfg_file, "r") as f:
            self.config = json.load(f)
            
        self.max_question_length = self.config.get("max_question_length", 100)
        self.max_answer_length = self.config.get("max_answer_length", 100)
        self.max_remark_length = self.config.get("max_remark_length", 100)
        self.max_improvement_length = self.config.get("max_improvement_length", 100)

        # Load tokenizers
        tokenizer_files = {
            'questions': 'tokenizer_questions.pkl',
            'answers': 'tokenizer_answers.pkl',
            'remarks': 'tokenizer_remarks.pkl',
            'improvements': 'tokenizer_improvements.pkl'
        }
        
        for name, filename in tokenizer_files.items():
            filepath = os.path.join(self.preprocessor_path, filename)
            if os.path.exists(filepath):
                with open(filepath, "rb") as f:
                    setattr(self, f'tokenizer_{name}', pickle.load(f))
                if name in ['remarks', 'improvements']:
                    setattr(self, f'has_{name}', True)
            else:
                if name in ['questions', 'answers']:
                    raise FileNotFoundError(f"{filename} is required but missing.")
                else:
                    print(f"Warning: {filename} not found. {name.capitalize()} will be ignored.")
                    setattr(self, f'tokenizer_{name}', getattr(self, 'tokenizer_answers'))

        # Load sequences
        seq_path = os.path.join(self.preprocessor_path, "sequences.npz")
        if not os.path.exists(seq_path):
            raise FileNotFoundError("sequences.npz not found in preprocessor_path!")
            
        data = np.load(seq_path)
        self.X_questions = data["questions"]
        self.X_answers = data["answers"]
        self.X_remarks = data.get("remarks", np.zeros((len(self.X_questions), self.max_remark_length), dtype=np.int32))
        self.X_improvements = data.get("improvements", np.zeros((len(self.X_questions), self.max_improvement_length), dtype=np.int32))

        print(f"Loaded sequences.npz → shapes:")
        print(f"  X_questions: {self.X_questions.shape}")
        print(f"  X_answers: {self.X_answers.shape}")
        print(f"  X_remarks: {self.X_remarks.shape}")
        print(f"  X_improvements: {self.X_improvements.shape}")

        # Check if remarks/improvements contain meaningful data
        self.has_remarks = np.any(self.X_remarks > 2)
        self.has_improvements = np.any(self.X_improvements > 2)

        # Create or load splits
        self._create_splits()
        
        # Compute vocab sizes
        self.vocab_size_questions = len(self.tokenizer_questions.word_index) + 1
        self.vocab_size_answers = len(self.tokenizer_answers.word_index) + 1
        self.vocab_size_remarks = len(self.tokenizer_remarks.word_index) + 1
        self.vocab_size_improvements = len(self.tokenizer_improvements.word_index) + 1

        print("Enhanced preprocessor data loaded successfully!")
        print(f"Max lengths → Q: {self.max_question_length}, A: {self.max_answer_length}")
        print(f"Vocab sizes → Q: {self.vocab_size_questions}, A: {self.vocab_size_answers}")
        print(f"Has remarks = {self.has_remarks}, Has improvements = {self.has_improvements}")

        # Prepare training data
        self._prepare_training_data()

    def _create_splits(self):
        """Create train/val/test splits with better stratification."""
        splits_path = os.path.join(self.preprocessor_path, "splits.pkl")
        if os.path.exists(splits_path):
            with open(splits_path, "rb") as f:
                self.splits_indices = pickle.load(f)
            print("Loaded existing splits from splits.pkl.")
        else:
            n = len(self.X_questions)
            indices = np.arange(n)
            np.random.seed(42)  # For reproducibility
            np.random.shuffle(indices)
            
            train_end = int(0.70 * n)
            val_end = int(0.85 * n)
            
            self.splits_indices = {
                "train": indices[:train_end],
                "val": indices[train_end:val_end],
                "test": indices[val_end:]
            }
            
            with open(splits_path, "wb") as f:
                pickle.dump(self.splits_indices, f)
            print(f"Created new 70/15/15 splits. Train={train_end}, Val={val_end-train_end}, Test={n-val_end}")

    def _prepare_training_data(self):
        """Prepare training data with enhanced decoder sequences."""
        def build_enhanced_decoder_sequences(all_seqs: np.ndarray, max_len: int):
            N = all_seqs.shape[0]
            dec_input = np.zeros((N, max_len), dtype=np.int32)
            dec_target = np.zeros((N, max_len), dtype=np.int32)
            
            for i, seq in enumerate(all_seqs):
                # Find actual sequence length (non-zero tokens)
                actual_len = int(np.sum(seq > 0))
                capped_len = min(actual_len, max_len - 1)
                
                # Decoder input: <START> + sequence
                dec_input[i, 0] = 1  # <START> token
                if capped_len > 0:
                    dec_input[i, 1:capped_len+1] = seq[:capped_len]
                
                # Decoder target: sequence + <END>
                dec_target[i, :capped_len] = seq[:capped_len]
                if capped_len < max_len:
                    dec_target[i, capped_len] = 2  # <END> token
                    
            return dec_input, dec_target

        self.splits = {"train": {}, "val": {}, "test": {}}
        
        for split_name in ["train", "val", "test"]:
            idx = self.splits_indices[split_name]
            
            # Get split data
            Q_split = self.X_questions[idx]
            A_split = self.X_answers[idx]
            R_split = self.X_remarks[idx] if self.has_remarks else np.zeros_like(A_split)
            I_split = self.X_improvements[idx] if self.has_improvements else np.zeros((len(idx), self.max_improvement_length), dtype=np.int32)
            
            # Pad sequences
            Qp = pad_sequences(Q_split, maxlen=self.max_question_length, padding="post", truncating="post")
            Ap = pad_sequences(A_split, maxlen=self.max_answer_length, padding="post", truncating="post")
            Rp = pad_sequences(R_split, maxlen=self.max_remark_length, padding="post", truncating="post") if self.has_remarks else np.zeros_like(Ap)
            
            # Build decoder sequences
            ans_dec_inp, ans_dec_tgt = build_enhanced_decoder_sequences(Ap, self.max_answer_length)
            rem_dec_inp, rem_dec_tgt = build_enhanced_decoder_sequences(Rp, self.max_remark_length) if self.has_remarks else (np.zeros_like(ans_dec_inp), np.zeros_like(ans_dec_tgt))
            
            self.splits[split_name] = {
                "questions": Qp,
                "answers": Ap,
                "answer_decoder_input": ans_dec_inp,
                "answer_decoder_target": ans_dec_tgt,
                "remarks": Rp,
                "remark_decoder_input": rem_dec_inp,
                "remark_decoder_target": rem_dec_tgt,
                "improvements": I_split
            }

        print("Enhanced training/validation/test data prepared successfully:")
        for s in ["train", "val", "test"]:
            print(f"  {s:5s} → {self.splits[s]['questions'].shape[0]} samples")


# Enable mixed precision for better performance
tf.keras.mixed_precision.set_global_policy('mixed_float16')


class SuperiorLSTMModel:
    """Enhanced LSTM model with improved architecture and training strategies."""
    
    def __init__(self, preprocessor: EnhancedLSTMQAPreprocessorLoader, llm_client: CybersecurityLLM = None, model_name="superior_cybersec_lstm"):
        self.preprocessor = preprocessor
        self.llm_client = llm_client
        self.model_name = model_name
        self.model = None
        self.encoder_model = None
        self.decoder_model = None
        
        # Enhanced hyperparameters for better performance
        self.embedding_dim = 512
        self.encoder_units = 512
        self.decoder_units = 512
        self.attention_units = 256
        self.dropout_rate = 0.2
        self.learning_rate = 0.0005
        self.batch_size = 32
        self.epochs = 50
        
        # Model directories
        self.model_dir = f"./models/{self.model_name}"
        self.checkpoint_dir = f"{self.model_dir}/checkpoints"
        self.logs_dir = f"{self.model_dir}/logs"
        
        # Create directories
        for dir_path in [self.model_dir, self.checkpoint_dir, self.logs_dir]:
            os.makedirs(dir_path, exist_ok=True)
            
        print(f"Initialized Superior LSTM Model: {self.model_name}")

    def build_model(self):
        """Build an enhanced seq2seq model with superior architecture."""
        print("Building superior seq2seq model with enhanced attention...")
        
        # === ENCODER ===
        encoder_inputs = Input(shape=(self.preprocessor.max_question_length,), name='encoder_input')
        
        # Enhanced embedding with positional encoding
        encoder_embedding = Embedding(
            input_dim=self.preprocessor.vocab_size_questions,
            output_dim=self.embedding_dim,
            mask_zero=True,
            embeddings_initializer='glorot_uniform',
            embeddings_regularizer=l2(0.00001),
            name='encoder_embedding'
        )(encoder_inputs)
        
        encoder_embedding = Dropout(self.dropout_rate / 2)(encoder_embedding)
        
        # Multi-layer bidirectional encoder
        encoder_lstm_1 = Bidirectional(
            LSTM(
                self.encoder_units,
                return_sequences=True,
                return_state=True,
                dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate / 2,
                kernel_regularizer=l2(0.00001),
                recurrent_regularizer=l2(0.00001)
            ),
            name='encoder_lstm_1'
        )
        encoder_outputs_1, fh1, fc1, bh1, bc1 = encoder_lstm_1(encoder_embedding)
        
        # Second encoder layer - FIXED: Use same units to match dimensions
        encoder_lstm_2 = Bidirectional(
            LSTM(
                self.encoder_units,  # Changed from self.encoder_units // 2 to self.encoder_units
                return_sequences=True,
                return_state=True,
                dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate / 2,
                kernel_regularizer=l2(0.00001),
                recurrent_regularizer=l2(0.00001)
            ),
            name='encoder_lstm_2'
        )
        encoder_outputs_2, fh2, fc2, bh2, bc2 = encoder_lstm_2(encoder_outputs_1)
        
        # Add residual connection - Now both outputs have the same shape
        encoder_outputs = Add(name='encoder_residual')([encoder_outputs_1, encoder_outputs_2])
        encoder_outputs = LayerNormalization(epsilon=1e-6, name='encoder_norm')(encoder_outputs)
        
        # Combine states for decoder initialization - FIXED: Match decoder units
        # Project concatenated states to match decoder units
        state_h_combined = Concatenate(name='encoder_state_h_concat')([fh2, bh2])
        state_c_combined = Concatenate(name='encoder_state_c_concat')([fc2, bc2])
        
        # Project to decoder dimensions
        state_h = Dense(self.decoder_units, name='state_h_projection')(state_h_combined)
        state_c = Dense(self.decoder_units, name='state_c_projection')(state_c_combined)
        encoder_states = [state_h, state_c]
        
        # === DECODER ===
        decoder_inputs = Input(shape=(self.preprocessor.max_answer_length,), name='decoder_input')
        
        decoder_embedding = Embedding(
            input_dim=self.preprocessor.vocab_size_answers,
            output_dim=self.embedding_dim,
            mask_zero=True,
            embeddings_initializer='glorot_uniform',
            embeddings_regularizer=l2(0.00001),
            name='decoder_embedding'
        )(decoder_inputs)
        
        decoder_embedding = Dropout(self.dropout_rate / 2)(decoder_embedding)
        
        # Decoder LSTM
        decoder_lstm = LSTM(
            self.decoder_units,
            return_sequences=True,
            return_state=True,
            dropout=self.dropout_rate,
            recurrent_dropout=self.dropout_rate / 2,
            kernel_regularizer=l2(0.00001),
            recurrent_regularizer=l2(0.00001),
            name='decoder_lstm'
        )
        decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
        
        # Enhanced multi-head attention
        attention = MultiHeadAttention(
            num_heads=8,
            key_dim=self.attention_units // 8,
            dropout=self.dropout_rate / 2,
            name='multi_head_attention'
        )
        attention_output = attention(
            query=decoder_outputs,
            value=encoder_outputs,
            key=encoder_outputs
        )
        
        # Combine decoder output with attention
        decoder_combined = Add(name='decoder_attention_add')([decoder_outputs, attention_output])
        decoder_combined = LayerNormalization(epsilon=1e-6, name='decoder_norm1')(decoder_combined)
        
        # Enhanced feed-forward network
        decoder_ffn1 = Dense(
            self.decoder_units * 4,
            activation='gelu',
            kernel_regularizer=l2(0.00001),
            name='decoder_ffn1'
        )(decoder_combined)
        decoder_ffn1 = Dropout(self.dropout_rate)(decoder_ffn1)
        
        decoder_ffn2 = Dense(
            self.decoder_units,
            activation='gelu',
            kernel_regularizer=l2(0.00001),
            name='decoder_ffn2'
        )(decoder_ffn1)
        decoder_ffn2 = Dropout(self.dropout_rate)(decoder_ffn2)
        
        # Second residual connection
        decoder_combined = Add(name='decoder_residual')([decoder_combined, decoder_ffn2])
        decoder_combined = LayerNormalization(epsilon=1e-6, name='decoder_norm2')(decoder_combined)
        
        # Output layer with label smoothing consideration
        decoder_outputs_final = Dense(
            self.preprocessor.vocab_size_answers,
            activation='softmax',
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            dtype='float32',
            name='output_layer'
        )(decoder_combined)
        
        # Build model
        self.model = Model(
            inputs=[encoder_inputs, decoder_inputs],
            outputs=decoder_outputs_final,
            name='superior_seq2seq_model'
        )
        
        # Enhanced optimizer with gradient clipping
        optimizer = AdamW(
            learning_rate=self.learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8,
            weight_decay=0.01,
            clipnorm=1.0
        )
        
        
        self.model.compile(
            optimizer=optimizer,
            loss=SparseCategoricalCrossentropy(from_logits=False),
            metrics=[SparseCategoricalAccuracy()],
            run_eagerly=False
        )
        
        print(f"Superior model built successfully! Parameters: {self.model.count_params():,}")
        self._build_inference_models()
        
        return self.model

    def _build_inference_models(self):
        """Build optimized inference models."""
        print("Building enhanced inference models...")
        
        # Encoder inference model
        encoder_inputs_inf = Input(shape=(self.preprocessor.max_question_length,), name='encoder_input_inf')
        
        # Get layers from main model
        encoder_embedding_layer = self.model.get_layer('encoder_embedding')
        encoder_lstm_1_layer = self.model.get_layer('encoder_lstm_1')
        encoder_lstm_2_layer = self.model.get_layer('encoder_lstm_2')
        encoder_residual_layer = self.model.get_layer('encoder_residual')
        encoder_norm_layer = self.model.get_layer('encoder_norm')
        
        # Forward pass
        encoder_embedded_inf = encoder_embedding_layer(encoder_inputs_inf)
        encoder_outputs_1_inf, fh1_inf, fc1_inf, bh1_inf, bc1_inf = encoder_lstm_1_layer(encoder_embedded_inf)
        encoder_outputs_2_inf, fh2_inf, fc2_inf, bh2_inf, bc2_inf = encoder_lstm_2_layer(encoder_outputs_1_inf)
        encoder_outputs_inf = encoder_residual_layer([encoder_outputs_1_inf, encoder_outputs_2_inf])
        encoder_outputs_inf = encoder_norm_layer(encoder_outputs_inf)
        
        state_h_inf = Concatenate(name='encoder_inf_state_h')([fh2_inf, bh2_inf])
        state_c_inf = Concatenate(name='encoder_inf_state_c')([fc2_inf, bc2_inf])
        
        self.encoder_model = Model(
            inputs=encoder_inputs_inf,
            outputs=[encoder_outputs_inf, state_h_inf, state_c_inf],
            name='encoder_model_inference'
        )
        
        decoder_inputs_inf = Input(shape=(1,), name='decoder_input_inf')
        decoder_state_input_h = Input(shape=(self.decoder_units,), name='decoder_state_h')
        decoder_state_input_c = Input(shape=(self.decoder_units,), name='decoder_state_c')
        encoder_outputs_input = Input(shape=(self.preprocessor.max_question_length, self.encoder_units), name='encoder_outputs_inf')
        
        # Get decoder layers
        decoder_embedding_layer = self.model.get_layer('decoder_embedding')
        decoder_lstm_layer = self.model.get_layer('decoder_lstm')
        attention_layer = self.model.get_layer('multi_head_attention')
        decoder_norm1_layer = self.model.get_layer('decoder_norm1')
        decoder_ffn1_layer = self.model.get_layer('decoder_ffn1')
        decoder_ffn2_layer = self.model.get_layer('decoder_ffn2')
        decoder_norm2_layer = self.model.get_layer('decoder_norm2')
        output_layer = self.model.get_layer('output_layer')
        
        # Forward pass
        decoder_embedded_inf = decoder_embedding_layer(decoder_inputs_inf)
        decoder_outputs_inf, state_h_inf2, state_c_inf2 = decoder_lstm_layer(
            decoder_embedded_inf, initial_state=[decoder_state_input_h, decoder_state_input_c]
        )
        
        attention_output_inf = attention_layer(
            query=decoder_outputs_inf,
            value=encoder_outputs_input,
            key=encoder_outputs_input
        )
        
        decoder_combined_inf = Add(name='decoder_attention_add_inf')([decoder_outputs_inf, attention_output_inf])
        decoder_combined_inf = decoder_norm1_layer(decoder_combined_inf)
        
        decoder_ffn_inf = decoder_ffn1_layer(decoder_combined_inf)
        decoder_ffn_inf = decoder_ffn2_layer(decoder_ffn_inf)
        
        decoder_combined_inf = Add(name='decoder_residual_inf')([decoder_combined_inf, decoder_ffn_inf])
        decoder_combined_inf = decoder_norm2_layer(decoder_combined_inf)
        
        decoder_outputs_final_inf = output_layer(decoder_combined_inf)
        
        self.decoder_model = Model(
            inputs=[decoder_inputs_inf, encoder_outputs_input, decoder_state_input_h, decoder_state_input_c],
            outputs=[decoder_outputs_final_inf, state_h_inf2, state_c_inf2],
            name='decoder_model_inference'
        )
        
        print("Enhanced inference models built successfully!")

    def custom_lr_schedule(self, epoch, lr):
        """Custom learning rate schedule for better convergence."""
        if epoch < 10:
            return lr
        elif epoch < 20:
            return lr * 0.9
        elif epoch < 30:
            return lr * 0.8
        else:
            return lr * 0.7

    def train_model(self, epochs=None, batch_size=None, use_advanced_training=True):
        """Train the model with advanced techniques."""
        if epochs is None:
            epochs = self.epochs
        if batch_size is None:
            batch_size = self.batch_size
            
        print(f"Training superior model for {epochs} epochs, batch size {batch_size}")
        
        if not hasattr(self, 'model') or self.model is None:
            self.build_model()
        
        # Enhanced callbacks
        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=8,
                restore_best_weights=True,
                verbose=1,
                min_delta=0.001
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.7,
                patience=4,
                min_lr=1e-7,
                verbose=1,
                min_delta=0.0005
            ),
            ModelCheckpoint(
                filepath=f"{self.checkpoint_dir}/best_model_{{epoch:02d}}_{{val_loss:.4f}}.h5",
                monitor="val_loss",
                save_best_only=True,
                verbose=1,
                save_weights_only=False
            ),
            CSVLogger(f"{self.logs_dir}/training_log.csv", append=True),
            TensorBoard(
                log_dir=self.logs_dir,
                histogram_freq=1,
                write_graph=True,
                write_images=True
            )
        ]
        
        if use_advanced_training:
            callbacks.append(
                LearningRateScheduler(self.custom_lr_schedule, verbose=1)
            )
        
        history = self.model.fit(
            [self.preprocessor.splits["train"]["questions"],
             self.preprocessor.splits["train"]["answer_decoder_input"]],
            self.preprocessor.splits["train"]["answer_decoder_target"],
            validation_data=(
                [self.preprocessor.splits["val"]["questions"],
                 self.preprocessor.splits["val"]["answer_decoder_input"]],
                self.preprocessor.splits["val"]["answer_decoder_target"]
            ),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
            shuffle=True
        )
        
        # Save final model
        final_path = f"{self.model_dir}/model_final.keras"
        self.model.save(final_path)
        print(f"Saved trained model to {final_path}")
        
        # Save training history
        history_path = f"{self.model_dir}/training_history.json"
        with open(history_path, 'w') as f:
            json.dump({
                'loss': [float(x) for x in history.history['loss']],
                'val_loss': [float(x) for x in history.history['val_loss']],
                'sparse_categorical_accuracy': [float(x) for x in history.history['sparse_categorical_accuracy']],
                'val_sparse_categorical_accuracy': [float(x) for x in history.history['val_sparse_categorical_accuracy']]
            }, f, indent=2)
        
        print(f"Training completed! Best validation accuracy: {max(history.history['val_sparse_categorical_accuracy']):.4f}")
        return history

    def save_model(self, path=None):
        """Save the complete model."""
        if path is None:
            path = f"{self.model_dir}/complete_model.keras"
        self.model.save(path)
        print(f"Model saved to {path}")

    def load_model(self, path=None):
        """Load a saved model."""
        if path is None:
            path = f"{self.model_dir}/complete_model.keras"
        self.model = load_model(path)
        self._build_inference_models()
        print(f"Model loaded from {path}")

    def beam_search_decode(self, encoder_outputs, encoder_states, beam_width=5, max_length=None):
        """Enhanced beam search for better text generation."""
        if max_length is None:
            max_length = self.preprocessor.max_answer_length
        
        # Initialize beam
        initial_state = encoder_states
        beams = [(0.0, [1], initial_state)]  # (score, sequence, state)
        completed_beams = []
        
        for step in range(max_length - 1):
            candidates = []
            
            for score, seq, state in beams:
                if seq[-1] == 2:  # End token
                    completed_beams.append((score, seq))
                    continue
                
                # Get predictions for current sequence
                decoder_input = np.array([[seq[-1]]])
                state_h, state_c = state
                
                predictions, new_state_h, new_state_c = self.decoder_model.predict([
                    decoder_input, encoder_outputs, 
                    np.expand_dims(state_h, 0), np.expand_dims(state_c, 0)
                ], verbose=0)
                
                # Get top beam_width predictions
                top_indices = np.argsort(predictions[0, 0, :])[-beam_width:]
                
                for idx in top_indices:
                    new_score = score + np.log(predictions[0, 0, idx] + 1e-8)
                    new_seq = seq + [int(idx)]
                    new_state = (new_state_h[0], new_state_c[0])
                    candidates.append((new_score, new_seq, new_state))
            
            # Select top beam_width candidates
            candidates.sort(key=lambda x: x[0], reverse=True)
            beams = candidates[:beam_width]
            
            if not beams:
                break
        
        # Add remaining beams to completed
        completed_beams.extend([(score, seq) for score, seq, _ in beams])
        
        if not completed_beams:
            return [1, 2]  # Return minimal sequence if nothing found
        
        # Return best sequence
        completed_beams.sort(key=lambda x: x[0] / len(x[1]), reverse=True)  # Normalize by length
        return completed_beams[0][1]

    def nucleus_sampling_decode(self, encoder_outputs, encoder_states, p=0.9, temperature=0.8, max_length=None):
        """Nucleus sampling for more diverse and coherent text generation."""
        if max_length is None:
            max_length = self.preprocessor.max_answer_length
        
        sequence = [1]  # Start with START token
        state_h, state_c = encoder_states
        
        for _ in range(max_length - 1):
            decoder_input = np.array([[sequence[-1]]])
            
            predictions, state_h, state_c = self.decoder_model.predict([
                decoder_input, encoder_outputs,
                np.expand_dims(state_h, 0), np.expand_dims(state_c, 0)
            ], verbose=0)
            
            # Apply temperature
            logits = predictions[0, 0, :] / temperature
            probs = tf.nn.softmax(logits).numpy()
            
            # Nucleus sampling
            sorted_indices = np.argsort(probs)[::-1]
            sorted_probs = probs[sorted_indices]
            cumsum_probs = np.cumsum(sorted_probs)
            
            # Find nucleus
            nucleus_size = np.searchsorted(cumsum_probs, p) + 1
            nucleus_indices = sorted_indices[:nucleus_size]
            nucleus_probs = sorted_probs[:nucleus_size]
            nucleus_probs = nucleus_probs / nucleus_probs.sum()
            
            # Sample from nucleus
            next_token = np.random.choice(nucleus_indices, p=nucleus_probs)
            sequence.append(int(next_token))
            
            if next_token == 2:  # End token
                break
        
        return sequence

    def enhanced_decode_sequence(self, input_seq, method='beam_search', **kwargs):
        """Enhanced decoding with multiple strategies for better output quality."""
        # Encode the input
        encoder_outputs, state_h, state_c = self.encoder_model.predict(input_seq, verbose=0)
        encoder_states = (state_h[0], state_c[0])
        
        if method == 'beam_search':
            decoded_sequence = self.beam_search_decode(
                encoder_outputs, encoder_states, 
                beam_width=kwargs.get('beam_width', 5)
            )
        elif method == 'nucleus_sampling':
            decoded_sequence = self.nucleus_sampling_decode(
                encoder_outputs, encoder_states,
                p=kwargs.get('p', 0.9),
                temperature=kwargs.get('temperature', 0.8)
            )
        else:
            # Fallback to greedy decoding
            decoded_sequence = self.greedy_decode(encoder_outputs, encoder_states)
        
        return decoded_sequence

    def greedy_decode(self, encoder_outputs, encoder_states):
        """Improved greedy decoding."""
        sequence = [1]  # Start token
        state_h, state_c = encoder_states
        
        for _ in range(self.preprocessor.max_answer_length - 1):
            decoder_input = np.array([[sequence[-1]]])
            
            predictions, state_h, state_c = self.decoder_model.predict([
                decoder_input, encoder_outputs,
                np.expand_dims(state_h, 0), np.expand_dims(state_c, 0)
            ], verbose=0)
            
            next_token = np.argmax(predictions[0, 0, :])
            sequence.append(int(next_token))
            
            if next_token == 2:  # End token
                break
        
        return sequence

    def post_process_answer(self, answer_text):
        """Post-process generated text for better grammar and coherence."""
        if not answer_text or answer_text.strip() == "":
            return "I need more information to provide a comprehensive answer."
        
        # Basic grammar fixes
        answer = answer_text.strip()
        
        # Capitalize first letter
        if answer and answer[0].islower():
            answer = answer[0].upper() + answer[1:]
        
        # Ensure proper sentence ending
        if answer and not answer.endswith(('.', '!', '?', ':')):
            answer += '.'
        
        # Fix common grammar issues
        grammar_fixes = {
            r'\bi\b': 'I',  # Capitalize I
            r'\s+': ' ',    # Multiple spaces to single space
            r'\s*([.!?])\s*': r'\1 ',  # Proper spacing after punctuation
            r'\s*,\s*': ', ',  # Proper spacing after commas
            r'\s*:\s*': ': ',  # Proper spacing after colons
            r'([.!?])\s*([a-z])': lambda m: m.group(1) + ' ' + m.group(2).upper(),  # Capitalize after sentence end
        }
        
        for pattern, replacement in grammar_fixes.items():
            if callable(replacement):
                answer = re.sub(pattern, replacement, answer)
            else:
                answer = re.sub(pattern, replacement, answer)
        
        # Remove incomplete sentences at the end
        sentences = re.split(r'[.!?]+', answer)
        if len(sentences) > 1 and sentences[-1].strip() and len(sentences[-1].strip().split()) < 3:
            answer = '.'.join(sentences[:-1]) + '.'
        
        return answer

    def predict_answer(self, question, use_llm_fallback=True, decoding_method='beam_search'):
        """Enhanced prediction with better text generation and LLM fallback."""
        try:
            # Preprocess question
            question_clean = question.strip().lower()
            question_seq = self.preprocessor.tokenizer_questions.texts_to_sequences([question_clean])
            question_padded = pad_sequences(
                question_seq, 
                maxlen=self.preprocessor.max_question_length, 
                padding='post'
            )
            
            # Check if cybersecurity-related and use LLM if available
            if (self.llm_client and 
                use_llm_fallback and 
                self.llm_client.is_cybersecurity_related(question)):
                
                try:
                    llm_answer = self.llm_client.generate_answer(question)
                    if llm_answer and "failed to generate" not in llm_answer.lower():
                        return f"[LLM] {llm_answer}"
                except Exception as e:
                    print(f"LLM fallback failed: {e}")
            
            # Generate answer using enhanced decoding
            decoded_sequence = self.enhanced_decode_sequence(
                question_padded, 
                method=decoding_method,
                beam_width=7,
                p=0.85,
                temperature=0.7
            )
            
            # Convert to text
            answer_words = []
            for token in decoded_sequence:
                if token == 0 or token == 1:  # PAD or START
                    continue
                elif token == 2:  # END
                    break
                else:
                    word = self.preprocessor.tokenizer_answers.index_word.get(token, '<UNK>')
                    if word != '<UNK>':
                        answer_words.append(word)
            
            raw_answer = ' '.join(answer_words)
            
            # Post-process for better grammar
            processed_answer = self.post_process_answer(raw_answer)
            
            # Quality check - if answer is too short or generic, try alternative method
            if (len(processed_answer.split()) < 5 or 
                processed_answer.lower() in ['the answer is', 'it is', 'this is']):
                
                if decoding_method != 'nucleus_sampling':
                    # Try nucleus sampling for more diverse output
                    return self.predict_answer(question, use_llm_fallback=False, decoding_method='nucleus_sampling')
                elif use_llm_fallback and self.llm_client:
                    # Last resort: try LLM anyway
                    try:
                        llm_answer = self.llm_client.generate_answer(question)
                        if llm_answer and "failed to generate" not in llm_answer.lower():
                            return f"[LLM] {llm_answer}"
                    except:
                        pass
            
            return processed_answer if processed_answer else "I need more context to provide a detailed answer."
            
        except Exception as e:
            print(f"Prediction error: {e}")
            if use_llm_fallback and self.llm_client:
                try:
                    return f"[LLM] {self.llm_client.generate_answer(question)}"
                except:
                    pass
            return "I apologize, but I encountered an error processing your question."

    def evaluate_model(self, split='test'):
        """Enhanced model evaluation with multiple metrics."""
        print(f"Evaluating model on {split} set...")
        
        questions = self.preprocessor.splits[split]['questions']
        answers = self.preprocessor.splits[split]['answer_decoder_target']
        
        # Standard evaluation
        loss, accuracy = self.model.evaluate(
            [questions, self.preprocessor.splits[split]['answer_decoder_input']],
            answers,
            batch_size=32,
            verbose=1
        )
        
        print(f"{split.capitalize()} Loss: {loss:.4f}")
        print(f"{split.capitalize()} Accuracy: {accuracy:.4f}")
        
        # BLEU score evaluation (sample-based)
        sample_size = min(100, len(questions))
        indices = np.random.choice(len(questions), sample_size, replace=False)
        
        bleu_scores = []
        for i in indices:
            question_text = self.sequence_to_text(questions[i], 'questions')
            true_answer = self.sequence_to_text(answers[i], 'answers')
            pred_answer = self.predict_answer(question_text, use_llm_fallback=False)
            
            # Simple BLEU-like score
            true_words = set(true_answer.lower().split())
            pred_words = set(pred_answer.lower().split())
            
            if len(pred_words) > 0:
                bleu = len(true_words.intersection(pred_words)) / len(pred_words.union(true_words))
                bleu_scores.append(bleu)
        
        avg_bleu = np.mean(bleu_scores) if bleu_scores else 0
        print(f"Average BLEU-like Score: {avg_bleu:.4f}")
        
        return {'loss': loss, 'accuracy': accuracy, 'bleu': avg_bleu}

    def sequence_to_text(self, sequence, tokenizer_type='answers'):
        """Convert sequence back to text."""
        if tokenizer_type == 'questions':
            tokenizer = self.preprocessor.tokenizer_questions
        else:
            tokenizer = self.preprocessor.tokenizer_answers
            
        words = []
        for token in sequence:
            if token == 0:  # PAD
                continue
            elif token == 1:  # START
                continue
            elif token == 2:  # END
                break
            else:
                word = tokenizer.index_word.get(token, '<UNK>')
                if word != '<UNK>':
                    words.append(word)
        
        return ' '.join(words)

    def generate_training_report(self):
        """Generate a comprehensive training report."""
        report_path = f"{self.model_dir}/training_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("=== ENHANCED CYBERSECURITY LSTM MODEL TRAINING REPORT ===\n\n")
            f.write(f"Model Name: {self.model_name}\n")
            f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("=== MODEL ARCHITECTURE ===\n")
            f.write(f"Embedding Dimension: {self.embedding_dim}\n")
            f.write(f"Encoder Units: {self.encoder_units}\n")
            f.write(f"Decoder Units: {self.decoder_units}\n")
            f.write(f"Attention Units: {self.attention_units}\n")
            f.write(f"Dropout Rate: {self.dropout_rate}\n")
            f.write(f"Learning Rate: {self.learning_rate}\n")
            f.write(f"Total Parameters: {self.model.count_params():,}\n\n")
            
            f.write("=== DATASET INFO ===\n")
            f.write(f"Max Question Length: {self.preprocessor.max_question_length}\n")
            f.write(f"Max Answer Length: {self.preprocessor.max_answer_length}\n")
            f.write(f"Question Vocab Size: {self.preprocessor.vocab_size_questions}\n")
            f.write(f"Answer Vocab Size: {self.preprocessor.vocab_size_answers}\n")
            f.write(f"Training Samples: {len(self.preprocessor.splits['train']['questions'])}\n")
            f.write(f"Validation Samples: {len(self.preprocessor.splits['val']['questions'])}\n")
            f.write(f"Test Samples: {len(self.preprocessor.splits['test']['questions'])}\n\n")
            
            # Evaluation results
            eval_results = self.evaluate_model('test')
            f.write("=== PERFORMANCE METRICS ===\n")
            f.write(f"Test Accuracy: {eval_results['accuracy']:.4f}\n")
            f.write(f"Test Loss: {eval_results['loss']:.4f}\n")
            f.write(f"BLEU Score: {eval_results['bleu']:.4f}\n\n")
            
            f.write("=== ENHANCEMENTS IMPLEMENTED ===\n")
            f.write("- Enhanced multi-layer bidirectional encoder\n")
            f.write("- Multi-head attention mechanism\n")
            f.write("- Residual connections and layer normalization\n")
            f.write("- Advanced regularization techniques\n")
            f.write("- Beam search and nucleus sampling decoding\n")
            f.write("- Post-processing for better grammar\n")
            f.write("- LLM fallback integration\n")
            f.write("- Custom learning rate scheduling\n")
            f.write("- Enhanced training callbacks\n")
        
        print(f"Training report saved to {report_path}")


class EnhancedCybersecurityQASystem:
    """Complete enhanced QA system with improved performance."""
    
    def __init__(self, preprocessor_path, groq_api_key=None, model_name="enhanced_cybersec_qa"):
        self.preprocessor = EnhancedLSTMQAPreprocessorLoader(preprocessor_path)
        self.llm_client = CybersecurityLLM(api_key=groq_api_key) if groq_api_key else None
        self.model = SuperiorLSTMModel(self.preprocessor, self.llm_client, model_name)
        
    def train(self, epochs=60, batch_size=24):
        """Train the enhanced model with optimized parameters."""
        print("Starting enhanced training process...")
        history = self.model.train_model(epochs=epochs, batch_size=batch_size, use_advanced_training=True)
        self.model.generate_training_report()
        return history
    
    def ask(self, question, method='beam_search'):
        """Ask a question and get an enhanced answer."""
        return self.model.predict_answer(question, decoding_method=method)
    
    def evaluate(self):
        """Evaluate the complete system."""
        return self.model.evaluate_model()
    
    def save_system(self, path=None):
        """Save the complete system."""
        self.model.save_model(path)
    
    def load_system(self, path=None):
        """Load a saved system."""
        self.model.load_model(path)


# Enhanced usage example and testing
def demonstrate_enhanced_system():
    """Demonstrate the enhanced system capabilities."""
    
    # Initialize system (replace with your actual paths)
    preprocessor_path = "./preprocessed_data"  # Update this path
    groq_api_key = "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"   # Replace with actual key
    
    try:
        # Create enhanced system
        qa_system = EnhancedCybersecurityQASystem(
            preprocessor_path=preprocessor_path,
            groq_api_key=groq_api_key,
            model_name="enhanced_cybersec_v2"
        )
        
        # Build and train model
        print("Building enhanced model...")
        qa_system.model.build_model()
        
        print("Starting enhanced training...")
        history = qa_system.train(epochs=50, batch_size=24)
        
        # Test different decoding methods
        test_questions = [
            "What is a firewall?",
            "How does encryption work?",
            "What are the types of malware?",
            "Explain network security best practices",
            "What is penetration testing?"
        ]
        
        print("\n=== TESTING ENHANCED SYSTEM ===")
        for question in test_questions:
            print(f"\nQ: {question}")
            
            # Test beam search
            answer_beam = qa_system.ask(question, method='beam_search')
            print(f"A (Beam Search): {answer_beam}")
            
            # Test nucleus sampling
            answer_nucleus = qa_system.ask(question, method='nucleus_sampling')
            print(f"A (Nucleus): {answer_nucleus}")
        
        print("\n=== EVALUATION RESULTS ===")
        results = qa_system.evaluate()
        print(f"Final Accuracy: {results['accuracy']:.4f}")
        print(f"BLEU Score: {results['bleu']:.4f}")
        
        qa_system.save_system()
        print("\nEnhanced system saved successfully!")
        
    except Exception as e:
        print(f"Error in demonstration: {e}")
        print("Please ensure you have the preprocessor data and valid API key.")


if __name__ == "__main__":
    demonstrate_enhanced_system()
    pass
        