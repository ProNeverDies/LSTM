import os
import json
import numpy as np
import pickle
import re
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Embedding, Dropout, LayerNormalization,
    Concatenate, Bidirectional
)
from tensorflow.keras.layers import Layer
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger, TensorBoard
)
import warnings
warnings.filterwarnings("ignore")
from groq import Groq

@register_keras_serializable()
class CustomAttentionLayer(Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.W_enc = Dense(units, use_bias=False, name="W_enc_proj")
        self.W_dec = Dense(units, use_bias=False, name="W_dec_proj")
        self.V = Dense(1, use_bias=False, name="attention_score")

    def call(self, inputs, mask=None):
        enc_out, dec_out = inputs
        enc_mask = None
        if mask is not None and mask[0] is not None:
            enc_mask = tf.cast(mask[0], tf.float32)

        enc_proj = self.W_enc(enc_out)
        dec_proj = self.W_dec(dec_out)

        enc_exp = tf.expand_dims(enc_proj, axis=1)
        dec_exp = tf.expand_dims(dec_proj, axis=2)

        score_tensor = self.V(tf.tanh(enc_exp + dec_exp))
        score_tensor = tf.squeeze(score_tensor, axis=-1)

        if enc_mask is not None:
            enc_mask_exp = tf.expand_dims(enc_mask, axis=1)
            score_tensor += (1.0 - enc_mask_exp) * -1e9

        attn_weights = tf.nn.softmax(score_tensor, axis=-1)
        context = tf.matmul(attn_weights, enc_out)
        return context

    def compute_mask(self, inputs, mask=None):
        return None

    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units})
        return config

class CybersecurityLLM:
    """
    Groq-LLM wrapper. We've expanded the keyword list so the LLM will answer across
    steganography, watermarking, forensics, pentesting, AI, OS, and more—NOT just pure cybersecurity.
    """
    def __init__(self, api_key=None, model_name="llama3-8b-8192"):
        self.api_key = api_key or os.getenv('GROQ_API_KEY')
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
        """
        Build a large regex of “whole‐word” patterns so that ANY of the below
        topics will cause the LLM to answer. This is much broader than just “cybersecurity.”
        """
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

        # Compile a single case‐insensitive regex
        pattern = r"(?i)(" + r"|".join(keywords) + r")"
        self._keyword_regex = re.compile(pattern)

    def is_cybersecurity_related(self, question: str) -> bool:
        """Return True if question matches any of the expanded keywords."""
        if not question or not question.strip():
            return False
        return bool(self._keyword_regex.search(question))

    def generate_answer(self, question: str, context: str = None) -> str:
        """
        Generate an answer using Groq LLM, regardless of topic as long as it matches one
        of our expanded keywords above.
        """
        try:
            prompt = f"Question: {question}"
            if context:
                prompt = f"Context: {context}\n\n{prompt}"

            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user",   "content": prompt}
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
            return "Sorry, the LLM failed to generate an answer."

    def _format_answer(self, answer: str) -> str:
        """
        A simple formatter to indent bulleted lines, trim to ~140 words, etc.
        """
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

# ─────────────────────────────────────────────────────────────────────────────
#  2) Q&A Preprocessor Loader (exactly as in Model_improved.pdf)
# ─────────────────────────────────────────────────────────────────────────────

class LSTMQAPreprocessorLoader:
    """
    Updated loader that expects:
      - preprocessor_path/config.json
      - preprocessor_path/sequences.npz      # must contain arrays 'questions','answers','remarks','improvements'
      - preprocessor_path/tokenizer_questions.pkl
      - preprocessor_path/tokenizer_answers.pkl
      - preprocessor_path/tokenizer_remarks.pkl
      - preprocessor_path/tokenizer_improvements.pkl
      - (optional) preprocessor_path/vocab_sizes.pkl  [not strictly needed if tokenizers exist]
    """
    def __init__(self, preprocessor_path: str):
        self.preprocessor_path = preprocessor_path
        self.has_remarks = False
        self.has_improvements = False
        self._load_preprocessor_data()

    def _load_preprocessor_data(self):
        print("Loading preprocessor data from:", self.preprocessor_path)
        # ── 1) config.json ────────────────────────────────────────────────────────
        cfg_file = os.path.join(self.preprocessor_path, "config.json")
        with open(cfg_file, "r") as f:
            self.config = json.load(f)

        # Read max lengths from config (must exist)
        self.max_question_length   = self.config.get("max_question_length", 100)
        self.max_answer_length     = self.config.get("max_answer_length",   100)
        self.max_remark_length     = self.config.get("max_remark_length",   100)
        self.max_improvement_length= self.config.get("max_improvement_length", 100)

        # ── 2) Load tokenizer_pickles ─────────────────────────────────────────────
        #   (raise an error if any is missing)
        tq_path = os.path.join(self.preprocessor_path, "tokenizer_questions.pkl")
        ta_path = os.path.join(self.preprocessor_path, "tokenizer_answers.pkl")
        tr_path = os.path.join(self.preprocessor_path, "tokenizer_remarks.pkl")
        ti_path = os.path.join(self.preprocessor_path, "tokenizer_improvements.pkl")

        if not os.path.exists(tq_path) or not os.path.exists(ta_path):
            raise FileNotFoundError("tokenizer_questions.pkl or tokenizer_answers.pkl missing.")

        with open(tq_path, "rb") as f:
            self.tokenizer_questions = pickle.load(f)
        with open(ta_path, "rb") as f:
            self.tokenizer_answers = pickle.load(f)

        # Remarks / Improvements tokenizers are optional; if missing, we just reuse the answer‐tokenizer
        if os.path.exists(tr_path):
            with open(tr_path, "rb") as f:
                self.tokenizer_remarks = pickle.load(f)
            self.has_remarks = True
        else:
            print("Warning: tokenizer_remarks.pkl not found. Remarks will be ignored in training.")
            self.tokenizer_remarks = self.tokenizer_answers

        if os.path.exists(ti_path):
            with open(ti_path, "rb") as f:
                self.tokenizer_improvements = pickle.load(f)
            self.has_improvements = True
        else:
            print("Warning: tokenizer_improvements.pkl not found. Improvements will be ignored in training.")
            self.tokenizer_improvements = self.tokenizer_answers

        # ── 3) Load integer‐sequence arrays from sequences.npz ─────────────────────
        seq_path = os.path.join(self.preprocessor_path, "sequences.npz")
        if not os.path.exists(seq_path):
            raise FileNotFoundError("sequences.npz not found in preprocessor_path!")

        data = np.load(seq_path)
        # Expect data.files to contain exactly: ['questions','answers','remarks','improvements']
        self.X_questions    = data["questions"]
        self.X_answers      = data["answers"]
        self.X_remarks      = data["remarks"]      if "remarks" in data else np.zeros((len(self.X_questions), self.max_remark_length), dtype=np.int32)
        self.X_improvements = data["improvements"] if "improvements" in data else np.zeros((len(self.X_questions), self.max_improvement_length), dtype=np.int32)

        # Report shapes
        print(f"Loaded sequences.npz → shapes:")
        print(f"  X_questions:    {self.X_questions.shape}")
        print(f"  X_answers:      {self.X_answers.shape}")
        print(f"  X_remarks:      {self.X_remarks.shape}")
        print(f"  X_improvements: {self.X_improvements.shape}")

        # Determine if there is any meaningful nonzero remark/improvement
        if np.any(self.X_remarks > 2):
            self.has_remarks = True
        else:
            self.has_remarks = False
            print("Remarks sequences exist but are all zeros → disabling remark‐training.")

        if np.any(self.X_improvements > 2):
            self.has_improvements = True
        else:
            self.has_improvements = False
            print("Improvements sequences exist but are all zeros → disabling improvement‐training.")

        # ── 4) Build train/val/test splits (70/15/15) if not already present ─────
        splits_path = os.path.join(self.preprocessor_path, "splits.pkl")
        if os.path.exists(splits_path):
            with open(splits_path, "rb") as f:
                self.splits_indices = pickle.load(f)
            print("Loaded existing splits from splits.pkl.")
        else:
            n = len(self.X_questions)
            indices = np.arange(n)
            np.random.shuffle(indices)
            train_end = int(0.70 * n)
            val_end   = int(0.85 * n)
            self.splits_indices = {
                "train": indices[:train_end],
                "val":   indices[train_end:val_end],
                "test":  indices[val_end:]
            }
            with open(splits_path, "wb") as f:
                pickle.dump(self.splits_indices, f)
            print(f"Created new 70/15/15 splits and saved to splits.pkl. Train={train_end}, Val={val_end-train_end}, Test={n-val_end}.")

        # Finally, compute vocab sizes
        self.vocab_size_questions    = len(self.tokenizer_questions.word_index) + 1
        self.vocab_size_answers      = len(self.tokenizer_answers.word_index) + 1
        self.vocab_size_remarks      = len(self.tokenizer_remarks.word_index) + 1
        self.vocab_size_improvements = len(self.tokenizer_improvements.word_index) + 1

        print("Preprocessor data loaded successfully!")
        print(f"Max lengths → Q: {self.max_question_length}, A: {self.max_answer_length}, "
              f"R: {self.max_remark_length}, I: {self.max_improvement_length}")
        print(f"Vocab sizes  → Q: {self.vocab_size_questions}, A: {self.vocab_size_answers}, "
              f"R: {self.vocab_size_remarks}, I: {self.vocab_size_improvements}")
        print(f"Has remarks = {self.has_remarks}, Has improvements = {self.has_improvements}")

        # Build the actual training arrays for each split
        self._prepare_training_data()

    def _prepare_training_data(self):
        """
        For each of the train/val/test splits, create the following dict:
          - 'questions'             : padded input sequences for questions
          - 'answers'               : padded target sequences for answers
          - 'answer_decoder_input'  : (shifted-right) input sequences for answer‐decoder
          - 'answer_decoder_target' : (shifted-left) target sequences for answer‐decoder
          - 'remarks'               : padded sequences for remarks (if has_remarks=True)
          - 'remark_decoder_input'  : (shifted-right) input sequences for remark‐decoder
          - 'remark_decoder_target' : (shifted-left) target sequences for remark‐decoder
          - 'improvements'          : raw improvement sequences (used later for retraining)
        """
        def build_decoder_sequences(all_seqs: np.ndarray, max_len: int):
            """
            Given an array shape=(N, max_len) of integer tokens (already padded),
            build a pair (dec_input, dec_target) each shape=(N, max_len):
              - dec_input[i, 0] = <start>=1
                dec_input[i, 1: capped_len+1] = all_seqs[i, :capped_len]
              - dec_target[i, :capped_len] = all_seqs[i, 1 : capped_len+1]
                dec_target[i, capped_len] = <end>=2
            Any trailing positions remain zero.
            """
            N = all_seqs.shape[0]
            dec_input  = np.zeros_like(all_seqs)
            dec_target = np.zeros_like(all_seqs)
            for i, seq in enumerate(all_seqs):
                # Count how many nonzero tokens are in seq
                actual_len = int(np.sum(seq > 0))
                capped_len = min(actual_len, max_len - 1)
                # Start token at position 0
                dec_input[i, 0] = 1
                if capped_len > 0:
                    dec_input[i, 1 : capped_len + 1] = seq[:capped_len]
                    dec_target[i,  : capped_len]     = seq[1 : capped_len + 1]
                # Place <end>=2 immediately after capped_len
                if capped_len < max_len:
                    dec_target[i, capped_len] = 2
            return dec_input, dec_target

        # Prepare an empty container for splits
        self.splits = {"train": {}, "val": {}, "test": {}}

        for split_name in ["train", "val", "test"]:
            idx = self.splits_indices[split_name]
            Q_split = self.X_questions[idx]
            A_split = self.X_answers[idx]
            R_split = self.X_remarks[idx] if self.has_remarks else np.zeros_like(A_split)
            I_split = self.X_improvements[idx] if self.has_improvements else np.zeros((len(idx), self.max_improvement_length), dtype=np.int32)

            # 1) Ensure proper padding to max lengths
            Qp = pad_sequences(Q_split, maxlen=self.max_question_length, padding="post", truncating="post")
            Ap = pad_sequences(A_split, maxlen=self.max_answer_length,   padding="post", truncating="post")
            Rp = pad_sequences(R_split, maxlen=self.max_remark_length,   padding="post", truncating="post") if self.has_remarks else np.zeros_like(Ap)

            # 2) Build decoder‐input/target for answers and remarks
            ans_dec_inp, ans_dec_tgt = build_decoder_sequences(Ap, self.max_answer_length)
            if self.has_remarks:
                rem_dec_inp, rem_dec_tgt = build_decoder_sequences(Rp, self.max_remark_length)
            else:
                rem_dec_inp = np.zeros_like(ans_dec_inp)
                rem_dec_tgt = np.zeros_like(ans_dec_tgt)

            # 3) Store into self.splits[split_name]
            self.splits[split_name] = {
                "questions": Qp,
                "answers": Ap,
                "answer_decoder_input":  ans_dec_inp,
                "answer_decoder_target": ans_dec_tgt,
                "remarks": Rp,
                "remark_decoder_input":  rem_dec_inp,
                "remark_decoder_target": rem_dec_tgt,
                "improvements": I_split
            }

        print("Training/validation/test data prepared successfully:")
        for s in ["train", "val", "test"]:
            print(f"  {s:5s} → questions: {self.splits[s]['questions'].shape[0]}, "
                  f"answers: {self.splits[s]['answers'].shape[0]}, "
                  f"remarks: {self.splits[s]['remarks'].shape[0]}")


# ─────────────────────────────────────────────────────────────────────────────
#  3) EnhancedReflectionLSTMModel (copied verbatim from Model_improved.pdf,
#     except for two tiny changes: (a) save/load in native Keras format; 
#     (b) a small fix to 'load_model' so HDF5‐based loading registers NotEqual).
# ─────────────────────────────────────────────────────────────────────────────

class EnhancedReflectionLSTMModel:
    """
    Enhanced Reflection LSTM Model with LLM Integration and feedback‐based retraining.
    """
    def __init__(self, preprocessor: LSTMQAPreprocessorLoader, llm_client: CybersecurityLLM, model_name: str = "enhanced_reflection_lstm"):
        self.preprocessor = preprocessor
        self.llm_client = llm_client
        self.model_name = model_name

        # These models will be created in `build_answer_model()`
        self.answer_model = None

        # In‐session feedback
        self.feedback_data     = []  # ALL Q&A + user rating / remarks
        self.llm_feedback_data = []  # Those Qs where LSTM was wrong → store LLM answer

        # Hyperparameters (default)
        self.embedding_dim = 128
        self.lstm_units    = 256
        self.dropout_rate  = 0.3
        self.learning_rate = 0.001
        self.batch_size    = 32
        self.epochs        = 30

        # Directory setup
        self.model_dir      = f"./models/{self.model_name}"
        self.checkpoint_dir = f"{self.model_dir}/checkpoints"
        self.logs_dir       = f"{self.model_dir}/logs"
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

        print(f"Initialized Enhanced Reflection LSTM Model: {self.model_name}")
        print("LLM Integration: Enabled")

    def build_answer_model(self):
        """Build the answer‐generation LSTM with custom attention and compile it."""
        print("Building answer generation model...")

        # 1) Encoder (question_input → embedding → Bi‐LSTM)
        question_input = Input(shape=(self.preprocessor.max_question_length,), name="question_input")
        question_embed = Embedding(
            input_dim=self.preprocessor.vocab_size_questions,
            output_dim=self.embedding_dim,
            mask_zero=True,
            name="question_embedding"
        )(question_input)

        encoder_lstm = Bidirectional(
            LSTM(self.lstm_units, return_sequences=True, return_state=True, dropout=self.dropout_rate, recurrent_dropout=self.dropout_rate),
            name="question_encoder"
        )
        enc_out, f_h, f_c, b_h, b_c = encoder_lstm(question_embed)
        state_h = Concatenate(name="enc_state_h")([f_h, b_h])
        state_c = Concatenate(name="enc_state_c")([f_c, b_c])
        encoder_states = [state_h, state_c]

        # 2) Decoder (answer_input → embedding → LSTM)
        answer_input = Input(shape=(self.preprocessor.max_answer_length,), name="answer_input")
        answer_embed = Embedding(
            input_dim=self.preprocessor.vocab_size_answers,
            output_dim=self.embedding_dim,
            mask_zero=True,
            name="answer_embedding"
        )(answer_input)

        answer_lstm = LSTM(
            self.lstm_units * 2,
            return_sequences=True,
            return_state=True,
            dropout=self.dropout_rate,
            recurrent_dropout=self.dropout_rate,
            name="answer_decoder"
        )
        answer_outputs, _, _ = answer_lstm(answer_embed, initial_state=encoder_states)

        # 3) Custom Attention (matching enc_out vs. answer_outputs)
        

        attention_layer = CustomAttentionLayer(units=self.lstm_units * 2)

        context_vector = attention_layer([enc_out, answer_outputs],
                                         mask=[question_embed._keras_mask, answer_embed._keras_mask])

        # 4) Combine, normalize, dropout
        combined = Concatenate(name="answer_combined")([answer_outputs, context_vector])
        combined_norm = LayerNormalization(name="answer_norm")(combined)
        combined_dp = Dropout(self.dropout_rate, name="answer_dropout")(combined_norm)

        # 5) Final dense + softmax
        dense1 = Dense(self.lstm_units, activation="relu", name="answer_dense_1")(combined_dp)
        dense1_norm = LayerNormalization(name="answer_dense_1_norm")(dense1)
        dense1_dp = Dropout(self.dropout_rate, name="answer_dense_1_dropout")(dense1_norm)

        answer_output = Dense(
            self.preprocessor.vocab_size_answers,
            activation="softmax",
            name="answer_output"
        )(dense1_dp)

        # 6) Build & compile
        self.answer_model = Model(inputs=[question_input, answer_input], outputs=answer_output, name="answer_model")

        optimizer = Adam(learning_rate=self.learning_rate, clipnorm=1.0)
        self.answer_model.compile(
            optimizer=optimizer,
            loss=SparseCategoricalCrossentropy(),
            metrics=[SparseCategoricalAccuracy()]
        )

        print(f"Answer model built successfully! Params: {self.answer_model.count_params():,}")
        return self.answer_model

    def predict_answer(self, question_text: str, max_length: int = None, temperature: float = 0.8) -> str:
        """Generate an LSTM answer to 'question_text' (greedy or temperature‐scaled sampling)."""
        if not self.answer_model:
            raise ValueError("Answer model not built or loaded!")

        if max_length is None:
            max_length = self.preprocessor.max_answer_length

        seq = self.preprocessor.tokenizer_questions.texts_to_sequences([question_text])
        if not seq or not seq[0]:
            return "Sorry, I couldn't understand the question."

        padded_q = pad_sequences(seq, maxlen=self.preprocessor.max_question_length, padding="post", truncating="post")
        generated = [1]  # <START>=1

        for step in range(max_length - 1):
            # Build a decoder_input that contains all tokens generated so far
            decoder_input = np.zeros((1, self.preprocessor.max_answer_length))
            for i, token in enumerate(generated):
                if i < self.preprocessor.max_answer_length:
                    decoder_input[0, i] = token

            preds = self.answer_model.predict([padded_q, decoder_input], verbose=0)
            next_pos = min(len(generated), preds.shape[1] - 1)
            token_probs = preds[0, next_pos, :]

            if temperature > 0:
                token_probs = token_probs / temperature
                token_probs = np.exp(token_probs) / np.sum(np.exp(token_probs))
                next_token = np.random.choice(len(token_probs), p=token_probs)
            else:
                next_token = np.argmax(token_probs)

            # Stop if <END>=2
            if next_token == 2:
                break
            # If it chose <PAD>=0, pick a fallback among top‐5 non‐zero
            if next_token == 0:
                sorted_idxs = np.argsort(token_probs)[::-1]
                found = False
                for idx in sorted_idxs[1:6]:
                    if idx not in (0, 2):
                        next_token = idx
                        found = True
                        break
                if not found:
                    break

            generated.append(next_token)
            if len(generated) >= max_length:
                break

        if len(generated) <= 1:
            return "I couldn't generate a proper response."

        try:
            ans_tokens = generated[1:]
            ans_text = self.preprocessor.tokenizer_answers.sequences_to_texts([ans_tokens])[0]
            return ans_text.strip() or "Empty response generated."
        except Exception as e:
            return f"Error converting tokens to text: {e}"

    def dual_answer_session(self):
        """
        Interactive loop:
          - Ask a question → get an LSTM answer + LLM answer
          - Ask user to rate LSTM answer → if 'no', store (Q, LSTM_ans, LLM_ans, user_feedback)
          - Save all feedback at the end of session
        """
        print("\n=== DUAL Q&A SESSION (LSTM + LLM) ===")
        print("Type 'quit' to exit; 'feedback' to see summary; otherwise ask any question.")
        session_entries = []

        while True:
            try:
                question = input("\nEnter your question: ").strip()
                if question.lower() == "quit":
                    break
                if question.lower() == "feedback":
                    self.show_feedback_summary()
                    continue
                if not question:
                    print("Please type a non‐empty question.")
                    continue

                print("\n[LSTM is thinking...]")
                t0 = datetime.now()
                lstm_ans = self.predict_answer(question, temperature=0.8)
                dt_lstm = (datetime.now() - t0).total_seconds()
                print(f"LSTM Answer (in {dt_lstm:.2f}s):\n\n{lstm_ans}")

                print("\n[LLM is thinking...]")
                t1 = datetime.now()
                if self.llm_client.is_cybersecurity_related(question):
                    llm_ans = self.llm_client.generate_answer(question)
                else:
                    llm_ans = "This question isn't cybersecurity‐related. I specialize in cybersecurity topics."
                dt_llm = (datetime.now() - t1).total_seconds()
                print(f"\nLLM Answer (in {dt_llm:.2f}s):\n\n{llm_ans}")

                # Collect feedback
                print("\n--- LSTM Feedback ---")
                rst = input("Is the LSTM answer correct? (yes/no/partial): ").strip().lower()
                rating = input("Rate LSTM answer quality (1–5): ").strip()
                remark = input("Specific feedback on LSTM answer: ").strip()

                # If LSTM was marked incorrect, store LLM answer for future retraining
                if rst in ("no", "incorrect", "wrong"):
                    print("Saving LLM answer as reference to retrain LSTM later...")
                    self.llm_feedback_data.append({
                        "timestamp": datetime.now().isoformat(),
                        "question": question,
                        "lstm_answer": lstm_ans,
                        "llm_reference_answer": llm_ans,
                        "user_remark": remark,
                        "lstm_rating": rating
                    })

                # Always store the session entry
                session_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "question": question,
                    "lstm_answer": lstm_ans,
                    "llm_answer": llm_ans,
                    "lstm_correct": rst,
                    "lstm_rating": rating,
                    "lstm_feedback": remark,
                    "lstm_time": dt_lstm,
                    "llm_time": dt_llm,
                    "is_cybersecurity_related": self.llm_client.is_cybersecurity_related(question)
                }
                session_entries.append(session_entry)
                self.feedback_data.append(session_entry)

                print("Feedback recorded. Ask another question or type 'quit'.")

            except KeyboardInterrupt:
                print("\nSession interrupted by user.")
                break
            except Exception as e:
                print(f"[Session ERROR] {e}")
                continue

        # End of session: save feedback to disk
        if session_entries:
            self._save_feedback(session_entries)
            print(f"Session ended. Collected {len(session_entries)} entries.")
            if self.llm_feedback_data:
                print(f" → {len(self.llm_feedback_data)} incorrect LSTM answers available for retraining.")
                choice = input("Would you like to prepare retraining data now? (y/n): ").strip().lower()
                if choice == "y":
                    self.prepare_llm_feedback_for_retraining()

        return session_entries

    def show_feedback_summary(self):
        """Print a brief summary of all collected feedback so far."""
        if not self.feedback_data:
            print("No feedback data collected yet.")
            return
        total = len(self.feedback_data)
        correct   = sum(1 for e in self.feedback_data if e.get("lstm_correct") == "yes")
        incorrect = sum(1 for e in self.feedback_data if e.get("lstm_correct") == "no")
        partial   = sum(1 for e in self.feedback_data if e.get("lstm_correct") == "partial")
        print("\n=== FEEDBACK SUMMARY ===")
        print(f"Total questions asked: {total}")
        print(f"  Correct LSTM answers:   {correct} ({correct/total*100:.1f}%)")
        print(f"  Incorrect LSTM answers: {incorrect} ({incorrect/total*100:.1f}%)")
        print(f"  Partial LSTM answers:   {partial} ({partial/total*100:.1f}%)")
        if self.llm_feedback_data:
            print(f"LLM references saved for retraining: {len(self.llm_feedback_data)}")
        print("\nMost recent feedback entries:")
        for idx, ent in enumerate(self.feedback_data[-5:], start=1):
            print(f"  {idx}. Q: {ent['question'][:50]}...  LSTM_correct: {ent['lstm_correct']}  Rating: {ent['lstm_rating']}")

    def _save_feedback(self, session_entries):
        """Write JSON files of all feedback and llm_feedback_data."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 1) All session entries
        sfile = f"{self.model_dir}/session_feedback_{timestamp}.json"
        with open(sfile, "w") as f:
            json.dump(session_entries, f, indent=2)
        print(f"Saved session feedback to: {sfile}")

        # 2) LLM feedback (only where LSTM was wrong)
        if self.llm_feedback_data:
            lfile = f"{self.model_dir}/llm_feedback_{timestamp}.json"
            with open(lfile, "w") as f:
                json.dump(self.llm_feedback_data, f, indent=2)
            print(f"Saved LLM feedback data to: {lfile}")

    def prepare_llm_feedback_for_retraining(self):
        """
        Convert self.llm_feedback_data → {questions, answers} lists + JSON file,
        ready for calling retrain_with_llm_feedback(...).
        """
        if not self.llm_feedback_data:
            print("No LLM feedback data to prepare.")
            return None

        print(f"Preparing {len(self.llm_feedback_data)} items for retraining...")
        retrain_list = []
        for ent in self.llm_feedback_data:
            retrain_list.append({
                "question": ent["question"],
                "correct_answer": ent["llm_reference_answer"]
            })

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file = f"{self.model_dir}/retraining_data_{timestamp}.json"
        with open(out_file, "w") as f:
            json.dump(retrain_list, f, indent=2)
        print(f"Saved retraining JSON to: {out_file}")

        questions = [x["question"] for x in retrain_list]
        answers   = [x["correct_answer"] for x in retrain_list]
        return {"questions": questions, "answers": answers, "retraining_file": out_file, "data": retrain_list}

    def retrain_with_llm_feedback(self, retraining_data=None, epochs=10):
        """
        Retrain the LSTM using the LLM’s reference answers (only those Qs where LSTM was wrong).
        """
        if retraining_data is None:
            retraining_data = self.prepare_llm_feedback_for_retraining()
        if not retraining_data:
            print("No retraining data available.")
            return False

        print(f"Retraining on {len(retraining_data['questions'])} examples for {epochs} epochs...")
        try:
            # 1) Convert to integer sequences
            q_seqs = self.preprocessor.tokenizer_questions.texts_to_sequences(retraining_data["questions"])
            a_seqs = self.preprocessor.tokenizer_answers.texts_to_sequences(retraining_data["answers"])

            X_retrain = pad_sequences(q_seqs, maxlen=self.preprocessor.max_question_length, padding="post")
            y_retrain = pad_sequences(a_seqs, maxlen=self.preprocessor.max_answer_length, padding="post")

            # 2) Build decoder input/target just like in preprocessor
            dec_inp_r = np.zeros_like(y_retrain)
            dec_tgt_r = np.zeros_like(y_retrain)
            for i, seq in enumerate(y_retrain):
                actual_len = np.sum(seq > 0)
                capped_len = min(actual_len, self.preprocessor.max_answer_length - 1)
                dec_inp_r[i, 0] = 1
                if capped_len > 0:
                    dec_inp_r[i, 1 : capped_len + 1] = seq[:capped_len]
                    dec_tgt_r[i, :capped_len]   = seq[1 : capped_len + 1]
                if capped_len < self.preprocessor.max_answer_length:
                    dec_tgt_r[i, capped_len] = 2

            # 3) Retrain
            callbacks = [
                EarlyStopping(monitor="loss", patience=3, restore_best_weights=True, verbose=1),
                ReduceLROnPlateau(monitor="loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1)
            ]
            hist = self.answer_model.fit(
                [X_retrain, dec_inp_r],
                dec_tgt_r,
                epochs=epochs,
                batch_size=16,
                callbacks=callbacks,
                verbose=1
            )

            # 4) Save the new model
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = f"{self.model_dir}/retrained_model_{ts}.keras"
            self.answer_model.save(out_path)
            print(f"Retrained model saved to: {out_path}")

            # 5) Also save history
            hist_file = f"{self.model_dir}/retraining_hist_{ts}.json"
            with open(hist_file, "w") as f:
                json.dump(hist.history, f, indent=2)
            print(f"Saved retraining history: {hist_file}")
            return True

        except Exception as e:
            print(f"[Retraining ERROR] {e}")
            return False

    def train_answer_model(self, epochs=None, batch_size=None):
        """Train a brand‐new answer model from scratch (using preprocessor.splits)."""
        if epochs is None: epochs = self.epochs
        if batch_size is None: batch_size = self.batch_size

        print(f"Training answer model for {epochs} epochs, batch size {batch_size}...")
        if not self.answer_model:
            self.build_answer_model()

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
            ModelCheckpoint(filepath=f"{self.checkpoint_dir}/best_answer_model.h5", monitor="val_loss", save_best_only=True, verbose=1),
            CSVLogger(f"{self.logs_dir}/train_log.csv", append=True),
            TensorBoard(log_dir=f"{self.logs_dir}/tensorboard", histogram_freq=1)
        ]

        hist = self.answer_model.fit(
            [self.preprocessor.splits["train"]["questions"], self.preprocessor.splits["train"]["answer_decoder_input"]],
            self.preprocessor.splits["train"]["answer_decoder_target"],
            validation_data=(
                [self.preprocessor.splits["val"]["questions"], self.preprocessor.splits["val"]["answer_decoder_input"]],
                self.preprocessor.splits["val"]["answer_decoder_target"]
            ),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        # Save the final model in native Keras format (.keras)
        out_file = f"{self.model_dir}/answer_model_final.keras"
        self.answer_model.save(out_file)
        print(f"Final answer model saved to: {out_file}")
        return hist

    def load_model(self, model_path=None):
        """Load a previously saved answer_model (native‐Keras .keras or .h5 + custom_objects)."""
        if model_path is None:
            model_path = f"{self.model_dir}/answer_model_final.keras"
            if not os.path.exists(model_path):
                # fallback to HDF5 if .keras not found
                model_path = f"{self.model_dir}/answer_model_final.h5"
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            return False

        try:
            if model_path.endswith(".h5"):
                # If legacy .h5, register NotEqual so loading does not break
                self.answer_model = load_model(
                    model_path,
                    custom_objects={"NotEqual": tf.math.not_equal}
                )
            else:
                self.answer_model = load_model(
                model_path,
                custom_objects={"CustomAttentionLayer": CustomAttentionLayer}
                )

            print(f"Loaded answer_model from: {model_path}")
            return True
        except Exception as e:
            print(f"[Load ERROR] {e}")
            return False

    def evaluate_model(self):
        """Evaluate on preprocessor.splits['test'] and print loss/accuracy."""
        if not self.answer_model:
            print("No model loaded/trained to evaluate.")
            return None
        print("Evaluating on test set...")
        loss, acc = self.answer_model.evaluate(
            [self.preprocessor.splits["test"]["questions"], self.preprocessor.splits["test"]["answer_decoder_input"]],
            self.preprocessor.splits["test"]["answer_decoder_target"],
            verbose=1
        )
        print(f"Test Loss: {loss:.4f}, Test Accuracy: {acc:.4f}")
        return {"test_loss": loss, "test_accuracy": acc}

    def quick_test_session(self, num_questions=3):
        """Prompt user for N test questions → show LSTM + LLM answers and ask if satisfied."""
        print(f"\n=== QUICK TEST SESSION ({num_questions} questions) ===")
        for i in range(num_questions):
            q = input(f"\nTest Question {i+1}: ").strip()
            if not q:
                continue
            print("\n[LSTM Answer:]")
            a_lstm = self.predict_answer(q)
            print(a_lstm)

            if self.llm_client.is_cybersecurity_related(q):
                print("\n[LLM Answer:]")
                a_llm = self.llm_client.generate_answer(q)
                print(a_llm)

            sats = input("\nSatisfied with LSTM answer? (y/n): ").strip().lower()
            if sats == "y":
                print("Great!")
            else:
                print("Consider adding this to feedback later.")

    def interactive_training_session(self):
        """
        Top-level “train or load then dual session, collect feedback, optional retrain”.
        """
        print("\n=== INTERACTIVE TRAINING & FEEDBACK SESSION ===")
        # 1) If no model loaded, ask user if they want to train
        if not self.answer_model:
            choice = input("No trained model found. Train a new model now? (y/n): ").strip().lower()
            if choice == "y":
                self.train_answer_model()
                print("Initial training completed.")
            else:
                print("Cannot proceed without a trained model.")
                return

        # 2) Run dual Q&A to collect feedback
        print("\nStarting dual Q&A session...")
        self.dual_answer_session()  # collects in self.feedback_data, self.llm_feedback_data

        # 3) If we have LLM feedback data, ask user if we should retrain
        if self.llm_feedback_data:
            print(f"\nYou have {len(self.llm_feedback_data)} incorrect LSTM answers to retrain on.")
            rchoice = input("Retrain the model using LLM feedback now? (y/n): ").strip().lower()
            if rchoice == "y":
                data = self.prepare_llm_feedback_for_retraining()
                if data:
                    e_input = input("How many epochs for retraining? (default 5): ").strip()
                    e_num = int(e_input) if e_input.isdigit() else 5
                    success = self.retrain_with_llm_feedback(data, epochs=e_num)
                    if success:
                        print("Retraining completed.")
                        t2 = input("Test the retrained model now? (y/n): ").strip().lower()
                        if t2 == "y":
                            self.quick_test_session()
                    else:
                        print("Retraining failed.")
        else:
            print("No incorrect LSTM answers found; feedback loop complete.")

        print("Interactive training session finished.")


# ─────────────────────────────────────────────────────────────────────────────
#  4) Top-level menu that ties everything together
# ─────────────────────────────────────────────────────────────────────────────

def main_menu():
    print("\n\n=== CYBERSEC LSTM + LLM FEEDBACK MENU ===")
    print("1) Train a brand-new LSTM model")
    print("2) Load existing LSTM model")
    print("3) Enter Dual Q&A session (LSTM + LLM + feedback)")
    print("4) View feedback summary + prepare retraining data")
    print("5) Retrain LSTM using LLM feedback")
    print("6) Exit")
    choice = input("Choose [1–6]: ").strip()
    return choice

def main():
    try:
        # 0) Initialize everything
        preprocessor_path = "./preprocessed_data"
        if not os.path.exists(preprocessor_path):
            print(f"Error: Preprocessor folder not found: {preprocessor_path}")
            return

        print("Loading preprocessor data...")
        preproc = LSTMQAPreprocessorLoader(preprocessor_path)

        # Initialize LLM client
        groq_key = os.getenv('GROQ_API_KEY') or "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"
        if not groq_key:
            print("Warning: GROQ_API_KEY not found → LLM answers will fail.")
        llm = CybersecurityLLM(api_key=groq_key)

        # Instantiate our enhanced model
        model = EnhancedReflectionLSTMModel(preprocessor=preproc, llm_client=llm, model_name="cybersecurity_reflection_lstm")

        while True:
            ch = main_menu()
            if ch == "1":
                # Train new model
                print("\n>> TRAIN A NEW MODEL")
                e = input("Enter number of epochs [default 30]: ").strip()
                epochs_to_run = int(e) if e.isdigit() else 30
                b = input("Enter batch size [default 32]: ").strip()
                bs = int(b) if b.isdigit() else 32
                model.train_answer_model(epochs=epochs_to_run, batch_size=bs)

            elif ch == "2":
                # Load existing model
                print("\n>> LOAD EXISTING MODEL")
                path_in = input("Enter path to model (.keras or .h5) [press Enter for default]: ").strip()
                path_in = path_in or None
                success = model.load_model(path_in)
                if success:
                    print("Model loaded successfully.")
                else:
                    print("Failed to load model.")

            elif ch == "3":
                # Dual Q&A session
                print("\n>> DUAL Q&A SESSION (and feedback)")
                if not model.answer_model:
                    print("No model loaded/trained yet.")
                    continue
                model.dual_answer_session()

            elif ch == "4":
                # View feedback summary + prepare retraining JSON
                print("\n>> VIEW FEEDBACK SUMMARY")
                model.show_feedback_summary()
                if model.llm_feedback_data:
                    print("You can now retrain on LLM feedback (choose option 5).")

            elif ch == "5":
                # Retrain on LLM feedback
                print("\n>> RETRAIN LSTM USING LLM FEEDBACK")
                data = model.prepare_llm_feedback_for_retraining()
                if data:
                    e2 = input("Epochs for retraining [default 5]: ").strip()
                    ep = int(e2) if e2.isdigit() else 5
                    ok = model.retrain_with_llm_feedback(data, epochs=ep)
                    if ok:
                        print("Retraining succeeded.")
                    else:
                        print("Retraining failed.")
                else:
                    print("No LLM feedback data available to retrain.")

            elif ch == "6":
                print("Exiting. Goodbye!")
                break

            else:
                print("Invalid choice. Please enter a number between 1 and 6.")

    except Exception as e:
        print(f"[MAIN ERROR] {e}")
        import traceback; traceback.print_exc()

if __name__ == "__main__":
    main()