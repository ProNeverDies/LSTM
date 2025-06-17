import os 
import json 
import numpy as np 
import pickle 
import re 
import tensorflow as tf 
from datetime import datetime 
from tensorflow.keras.models import load_model as _lm 
from tensorflow.keras.models import Model, load_model 
from tensorflow.keras.layers import ( 
    Input, LSTM, Dense, Embedding, Dropout, 
    LayerNormalization, Add, Concatenate, Bidirectional, 
    MultiHeadAttention 
) 
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.losses import SparseCategoricalCrossentropy 
from tensorflow.keras.metrics import SparseCategoricalAccuracy 
from tensorflow.keras.preprocessing.sequence import pad_sequences 
from tensorflow.keras.callbacks import ( 
    EarlyStopping, ReduceLROnPlateau, 
    ModelCheckpoint, CSVLogger, TensorBoard 
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
            - Cybersecurity fundamentals, best practices, and threat detection,all the concepts related to cybersecurity
            - Information security principles, risk management, and compliance
            - Data privacy & protection regulations 
            - Malware analysis and reverse engineering 
            - Computer networking and network security 
            - Operating system security & design 
            - Artificial intelligence security implications and AI concepts 
            - Digital forensics techniques and incident response 
            - Steganography & watermarking methods 
            - Cryptography (symmetric, asymmetric, hashing, PKI, etc.),different types of hash functions
            - Penetration testing and vulnerability management 
            - DevSecOps, cloud security and container security 
            - General technical questions on programming, OS, and computer science 
 
            Provide accurate, concise, educational responses. Keep answers under 150 words and format 
            clearly for learning. 
        """ 
        self._compile_keyword_pattern() 
 
    def _compile_keyword_pattern(self): 
        """ 
        Build a single case-insensitive regex that triggers the LLM whenever ANY of 
        the following keywords appear. 
        """ 
        keywords = [ 
            # Core Cybersecurity & Info-Sec 
            r"\bcybersecurity\b", r"\bsecurity\b", r"\binformation security\b", 
            r"\binfosec\b", r"\bdata protection\b", r"\bdata privacy\b", 
            r"\bprivacy\b", r"\bcompliance\b", r"\brisk management\b", 
            r"\bdisaster recovery\b", 
 
            # Malware & Forensics 
            r"\bmalware\b", r"\bvirus\b", r"\btrojan\b", r"\bransomware\b", 
            r"\bspyware\b", r"\brootkit\b", r"\bworm\b", r"\badware\b", 
            r"\bbotnet\b", r"\breverse engineering\b", r"\bforensics\b", 
            r"\bdigital forensics\b", r"\bincident response\b", r"\bthreat hunting\b", 
 
            # Steganography & Watermarking 
            r"\bsteganography\b", r"\bwatermarking\b", r"\bdct\b", r"\bdwt\b", 
            r"\baudio steganography\b", r"\btransform-domain steganography\b", 
 
            # Penetration Testing & Vulnerability Management 
            r"\bpenetration testing\b", r"\bpenetest\b", r"\bvulnerability management\b", 
            r"\bexploit\b", r"\bpatching\b", r"\bbug bounty\b", r"\bred teaming\b", 
            r"\bblue team\b", r"\bpurple team\b", 
 
            # Networking & Infrastructure 
            r"\bnetwork security\b", r"\bcomputer networks\b", r"\bfirewall\b", 
            r"\bids\b", r"\bips\b", r"\bvpn\b", r"\bproxy\b", r"\bcloud security\b", 
            r"\bcontainer security\b", r"\bkubernetes security\b", r"\bdevsecops\b", 
            r"\binfrastructure as code\b", r"\bcloud misconfigurations\b", 
 
            # Authentication & Access Control 
            r"\bauthentication\b", r"\bauthorization\b", r"\bmfa\b", r"\b2fa\b", 
            r"\bsso\b", r"\boauth\b", r"\bsaml\b", r"\bidentity management\b", 
            r"\bleast privilege\b", 
 
            # Cryptography & PKI 
            r"\bencryption\b", r"\bdecryption\b", r"\bhashing\b", r"\bssl\b", r"\btls\b", 
            r"\bcert(ificate)?\b", r"\bpk\b", r"\brsa\b", r"\baes\b", r"\bkey exchange\b", 
            r"\bpublic key\b", r"\bprivate key\b", r"\bdigital signature\b", r"\belliptic curve\b", 
            r"\bgnupg\b", r"\bpost-quantum cryptography\b", r"\bquantum encryption\b", 
 
            # Operating Systems & Virtualization 
            r"\boperating system\b", r"\bwindows security\b", r"\blinux security\b", 
            r"\bros security\b", r"\bmemory forensics\b", 
 
            # Artificial Intelligence & ML 
            r"\bartificial intelligence\b", r"\bai security\b", r"\bmachine learning\b", 
            r"\bdeep learning\b", r"\bneural networks\b", r"\bml security\b", 
            r"\bdeepfake detection\b", r"\bblockchain security\b", 
 
            # Database & Application Security 
            r"\bsql injection\b", r"\bxss\b", r"\bcsrf\b", r"\bclickjacking\b", 
            r"\bdirectory traversal\b", r"\bsession hijacking\b", r"\btoken manipulation\b", 
            r"\binput validation\b", r"\brate limiting\b", 
 
            # Social Engineering & Threats 
            r"\bphishing\b", r"\bsmishing\b", r"\bvishing\b", r"\bsocial engineering\b", 
            r"\bbrute force\b", r"\bcredential stuffing\b", r"\bddos\b", r"\bdos\b", 
            r"\bwatering hole\b", r"\bman in the middle\b", r"\beavesdropping\b", 
 
            # Standards, Regulations & Compliance 
            r"\bgdpr\b", r"\bhHIPAA\b", r"\bpci dss\b", r"\biso 27001\b", 
            r"\bnist\b", r"\bndpr\b", r"\bcisa\b", r"\bcmmc\b", r"\bsoc 2\b", r"\bccpa\b", 
 
            # Miscellaneous / General Tech (Programming, CS foundational topics) 
            r"\bprogramming\b", r"\bdata structures\b", r"\balgorithms\b", 
            r"\boperating systems\b", r"\bcompiler\b", r"\bcomputer architecture\b", 
            r"\bnumerical methods\b", r"\bcloud computing\b", r"\bdevops\b", 
            r"\bapi development\b", r"\bdatabase design\b", r"\bsoftware engineering\b", 
        ] 
 
        pattern = r"(?i)(" + r"|".join(keywords) + r")" 
        self._keyword_regex = re.compile(pattern) 
 
    def is_cybersecurity_related(self, question: str) -> bool: 
        """Return True if the question matches any of our expanded keywords.""" 
        if not question or not question.strip(): 
            return False 
        return bool(self._keyword_regex.search(question)) 
 
    def generate_answer(self, question: str, context: str = None) -> str: 
        """ 
        Generate an answer using Groq LLM, as long as the question matches our keywords. 
        Otherwise, respond with a fallback message. 
        """ 
        try: 
            prompt = f"Question: {question}"  
            if context: 
                prompt = f"Context: {context}\n\n{prompt}" 
 
            chat_completion = self.client.chat.completions.create( #Sends req to the llm using groq client 
                messages=[ 
                    {"role": "system", "content": self.system_prompt},#Used to define the tone of the model 
                    {"role": "user",   "content": prompt} 
                ], 
                model=self.model_name, 
                max_tokens=self.max_tokens, 
                temperature=0.7, #More temperature = more randomness 
                top_p=0.9, #Top 90% tokens to generate answer whose cummulative probability is 90%
                stream=False, #Answer is produced at once,not streamed 
            ) 
            answer = chat_completion.choices[0].message.content.strip() 
            return self._format_answer(answer) 
        except Exception as e: 
            print(f"[LLM ERROR] {e}") 
            return "Sorry, the LLM failed to generate an answer." 
 
    def _format_answer(self, answer: str) -> str: 
        """ 
        Simple formatter to indent bullet points, remove extra blank lines, and truncate around 140-150 words. 
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

class LSTMQAPreprocessorLoader: 
    """ 
    Loader that loads: 
      - preprocessor_path/config.json 
      - preprocessor_path/sequences.npz      # must contain arrays 
'questions','answers','remarks','improvements' 
      - preprocessor_path/tokenizer_questions.pkl 
      - preprocessor_path/tokenizer_answers.pkl 
      - preprocessor_path/tokenizer_remarks.pkl  
      - preprocessor_path/tokenizer_improvements.pkl  
      - preprocessor_path/vocab_sizes.pkl    # Optional 
    """ 
    def __init__(self, preprocessor_path: str): 
        self.preprocessor_path = preprocessor_path 
        self.has_remarks = False    #First entry has no remarks and improvements 
        self.has_improvements = False 
        self._load_preprocessor_data() 
    
    def _load_preprocessor_data(self): 
        print("Loading preprocessor data from:", self.preprocessor_path) 
         
        cfg_file = os.path.join(self.preprocessor_path, "config.json") 
        with open(cfg_file, "r") as f: 
            self.config = json.load(f) 
        
        self.max_question_length    = self.config.get("max_question_length", 100) 
        self.max_answer_length      = self.config.get("max_answer_length", 100) 
        self.max_remark_length      = self.config.get("max_remark_length", 100) 
        self.max_improvement_length = self.config.get("max_improvement_length", 100) 
 
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
 
        if os.path.exists(tr_path): 
            with open(tr_path, "rb") as f: 
                self.tokenizer_remarks = pickle.load(f) 
            self.has_remarks = True 
        else: 
            print("Warning: tokenizer_remarks.pkl not found. Remarks will be ignored.") 
            self.tokenizer_remarks = self.tokenizer_answers #Protects from non type error 
             
        if os.path.exists(ti_path): 
            with open(ti_path, "rb") as f: 
                self.tokenizer_improvements = pickle.load(f) 
            self.has_improvements = True 
        else: 
            print("Warning: tokenizer_improvements.pkl not found. Improvements will be ignored.") 
            self.tokenizer_improvements = self.tokenizer_answers #Protects from non type and attribute error 
 
        seq_path = os.path.join(self.preprocessor_path, "sequences.npz") 
        if not os.path.exists(seq_path): 
            raise FileNotFoundError("sequences.npz not found in preprocessor_path!") 
 
        data = np.load(seq_path) 
        self.X_questions    = data["questions"] 
        self.X_answers      = data["answers"] 
        self.X_remarks      = data["remarks"]      if "remarks" in data else np.zeros((len(self.X_questions), self.max_remark_length), dtype=np.int32) 
        self.X_improvements = data["improvements"] if "improvements" in data else np.zeros((len(self.X_questions), self.max_improvement_length), dtype=np.int32) 

        print(f"Loaded sequences.npz -> shapes:") 
        print(f"  X_questions:    {self.X_questions.shape}") 
        print(f"  X_answers:      {self.X_answers.shape}") 
        print(f"  X_remarks:      {self.X_remarks.shape}") 
        print(f"  X_improvements: {self.X_improvements.shape}") 
 
        #Determine if remarks/improvements are meaningful i.e they contain non-zero values
        if np.any(self.X_remarks > 2): 
            self.has_remarks = True 
        else: 
            self.has_remarks = False 
            print("Remarks exist but are all zeros i.e disabling remark-training.") 
 
        if np.any(self.X_improvements > 2): 
            self.has_improvements = True 
        else: 
            self.has_improvements = False 
            print("Improvements exist but are all zeros i.e disabling improvement-training.") 
 
        #Build train/val/test splits (70/15/15) if not present 
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

        self.vocab_size_questions    = len(self.tokenizer_questions.word_index) + 1 
        self.vocab_size_answers      = len(self.tokenizer_answers.word_index) + 1 
        self.vocab_size_remarks      = len(self.tokenizer_remarks.word_index) + 1 
        self.vocab_size_improvements = len(self.tokenizer_improvements.word_index) + 1 
 
        print("Preprocessor data loaded successfully!") 
        print(f"Max lengths -> Q: {self.max_question_length}, A: {self.max_answer_length}, R: {self.max_remark_length}, I: {self.max_improvement_length}") 
        print(f"Vocab sizes -> Q: {self.vocab_size_questions}, A: {self.vocab_size_answers}, R: {self.vocab_size_remarks}, I: {self.vocab_size_improvements}") 
        print(f"Has remarks = {self.has_remarks}, Has improvements = {self.has_improvements}") 
 
        self._prepare_training_data() 
 
    def _prepare_training_data(self): 
        """ 
        For each split, build: 
          - questions (padded) 
          - answers (padded) 
          - answer_decoder_input  (shifted-right) 
          - answer_decoder_target (shifted-left) 
          - remarks & improvements (if available) 
        """ 
        def build_decoder_sequences(all_seqs: np.ndarray, max_len: int): 
            N          = all_seqs.shape[0]       #Counts number of sequences in the batch
            dec_input  = np.zeros_like(all_seqs) 
            dec_target = np.zeros_like(all_seqs) 
            for i, seq in enumerate(all_seqs): 
                actual_len = int(np.sum(seq > 0)) 
                capped_len = min(actual_len, max_len - 1) #fits the length to the decorder input

                dec_input[i, 0] = 1  #<START> token 
                if capped_len > 0: 
                    dec_input[i, 1:capped_len+1] = seq[:capped_len] #Fills decorder input with original tokens shifted right
                    dec_target[i, :capped_len] = seq[1:capped_len+1] #Fills decorder targets with original tokens shifted left
 
                if capped_len < max_len: 
                    dec_target[i, capped_len] = 2  #<END> token 
                    
            return dec_input, dec_target 
 
        self.splits = {"train": {}, "val": {}, "test": {}} 
 
        for split_name in ["train", "val", "test"]: 
            idx     = self.splits_indices[split_name] 
            Q_split = self.X_questions[idx] 
            A_split = self.X_answers[idx] 
            R_split = self.X_remarks[idx] if self.has_remarks else np.zeros_like(A_split) 
            I_split = self.X_improvements[idx] if self.has_improvements else np.zeros((len(idx), self.max_improvement_length), dtype=np.int32) 
 
            Qp = pad_sequences(Q_split, maxlen=self.max_question_length, padding="post", truncating="post") 
            Ap = pad_sequences(A_split, maxlen=self.max_answer_length, padding="post", truncating="post") 
            Rp = pad_sequences(R_split, maxlen=self.max_remark_length, padding="post", truncating="post") if self.has_remarks else np.zeros_like(Ap) 
 
            ans_dec_inp, ans_dec_tgt = build_decoder_sequences(Ap, self.max_answer_length) 
            if self.has_remarks: 
                rem_dec_inp, rem_dec_tgt = build_decoder_sequences(Rp, self.max_remark_length) 
            else: 
                rem_dec_inp = np.zeros_like(ans_dec_inp) 
                rem_dec_tgt = np.zeros_like(ans_dec_tgt) 

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
            print(f"  {s:5s} -> questions: {self.splits[s]['questions'].shape[0]}, " 
                  f"answers: {self.splits[s]['answers'].shape[0]}, " 
                  f"remarks: {self.splits[s]['remarks'].shape[0]}") 

tf.keras.mixed_precision.set_global_policy('mixed_float16') 
 
class ImprovedLSTMModel: 
    def __init__(self, preprocessor: LSTMQAPreprocessorLoader, llm_client: CybersecurityLLM = None, 
model_name="cybersec_lstm_v2"): 
        self.preprocessor = preprocessor 
        self.llm_client = llm_client 
        self.model_name = model_name 
 
        self.model = None 
        self.encoder_model = None 
        self.decoder_model = None 
 
        self.embedding_dim = 512 
        self.encoder_units = 1024 
        self.decoder_units = 1024 
        self.attention_units = 512 
        self.dropout_rate = 0.3 
        self.learning_rate = 0.001 
        self.batch_size = 16 
        self.epochs = 100 
 
        self.model_dir = f"./models/{self.model_name}" 
        self.checkpoint_dir = f"{self.model_dir}/checkpoints" 
        self.logs_dir = f"{self.model_dir}/logs" 
        os.makedirs(self.model_dir, exist_ok=True) 
        os.makedirs(self.checkpoint_dir, exist_ok=True) 
        os.makedirs(self.logs_dir, exist_ok=True) 
 
        print(f"Initialized Improved LSTM Model: {self.model_name}") 
 
    def build_model(self): 
        print("Building improved seq2seq model with attention...") 
 
        encoder_inputs = Input( 
            shape=(self.preprocessor.max_question_length,), 
            name='encoder_input' 
        ) 
        
        encoder_embedding = Embedding( 
            input_dim=self.preprocessor.vocab_size_questions, 
            output_dim=self.embedding_dim, 
            mask_zero=True, 
            embeddings_initializer='glorot_uniform', 
            embeddings_regularizer=l2(0.0001), 
            name='encoder_embedding' 
        )(encoder_inputs) 
         
        encoder_embedding = Dropout(self.dropout_rate / 2)(encoder_embedding) #Protects from losing too much semantic information 
            
        encoder_lstm_1 = Bidirectional( 
            LSTM( 
                self.encoder_units, 
                return_sequences=True, 
                return_state=True, 
                dropout=self.dropout_rate, 
                recurrent_dropout=self.dropout_rate, 
                kernel_regularizer=l2(0.0001), 
                recurrent_regularizer=l2(0.0001) 
            ), 
            name='encoder_lstm_1' 
        ) 
        encoder_outputs_1, fh1, fc1, bh1, bc1 = encoder_lstm_1(encoder_embedding) 
         
        encoder_lstm_2 = Bidirectional( 
            LSTM( 
                self.encoder_units // 2, 
                return_sequences=True, 
                return_state=True, 
                dropout=self.dropout_rate, 
                recurrent_dropout=self.dropout_rate, 
                kernel_regularizer=l2(0.0001), 
                recurrent_regularizer=l2(0.0001) 
            ), 
            name='encoder_lstm_2' 
        ) 
        encoder_outputs, fh2, fc2, bh2, bc2 = encoder_lstm_2(encoder_outputs_1) 
         
        state_h = Concatenate(name='encoder_state_h')([fh2, bh2]) 
        state_c = Concatenate(name='encoder_state_c')([fc2, bc2]) 
        encoder_states = [state_h, state_c] 
 
        decoder_inputs = Input( 
            shape=(self.preprocessor.max_answer_length,), 
            name='decoder_input' 
        ) 
         
        decoder_embedding = Embedding( 
            input_dim=self.preprocessor.vocab_size_answers, 
            output_dim=self.embedding_dim, 
            mask_zero=True, 
            embeddings_initializer='glorot_uniform', 
            embeddings_regularizer=l2(0.0001), 
            name='decoder_embedding' 
        )(decoder_inputs) 
         
        decoder_embedding = Dropout(self.dropout_rate / 2)(decoder_embedding) 
 
        decoder_lstm = LSTM( 
            self.decoder_units, 
            return_sequences=True, 
            return_state=True, 
            dropout=self.dropout_rate, 
            recurrent_dropout=self.dropout_rate, 
            kernel_regularizer=l2(0.0001), 
            recurrent_regularizer=l2(0.0001), 
            name='decoder_lstm' 
        ) 
        decoder_outputs, _, _ = decoder_lstm( 
            decoder_embedding, 
            initial_state=encoder_states 
        ) 
 
        attention = MultiHeadAttention( 
            num_heads=16, 
            key_dim=self.attention_units // 16, 
            dropout=self.dropout_rate / 2, 
            name='attention' 
        ) 
        attention_output = attention( 
            query=decoder_outputs, 
            value=encoder_outputs, 
            key=encoder_outputs 
        ) 
 
        decoder_combined = Add(name='decoder_attention_add')([decoder_outputs, attention_output]) 
        decoder_combined = LayerNormalization(epsilon=1e-6, name='decoder_norm')(decoder_combined) 
         
        decoder_ffn = Dense( 
            self.decoder_units * 2, 
            activation='gelu',              #Gelu was getting results 
            kernel_regularizer=l2(0.0001), 
            name='decoder_ffn1' 
        )(decoder_combined) 
        decoder_ffn = Dropout(self.dropout_rate)(decoder_ffn) 
        decoder_ffn = Dense( 
            self.decoder_units, 
            activation='gelu', 
            kernel_regularizer=l2(0.0001), 
            name='decoder_ffn2' 
        )(decoder_ffn) 
        decoder_ffn = Dropout(self.dropout_rate)(decoder_ffn) 
         
        decoder_combined = Add(name='decoder_residual')([decoder_combined, decoder_ffn]) 
        decoder_combined = LayerNormalization(epsilon=1e-6, name='decoder_norm2')(decoder_combined) 
        
        decoder_outputs_final = Dense( 
            self.preprocessor.vocab_size_answers, 
            activation='softmax', 
            kernel_initializer='glorot_uniform', 
            bias_initializer='zeros', 
            dtype='float32', 
            name='output_layer' 
        )(decoder_combined) 
 
        self.model = Model( 
            inputs=[encoder_inputs, decoder_inputs], 
            outputs=decoder_outputs_final, 
            name='seq2seq_model' 
        ) 
         
        optimizer = Adam( 
            learning_rate=self.learning_rate, 
            beta_1=0.9, 
            beta_2=0.999, 
            epsilon=1e-8, 
            clipnorm=1.0 
        ) 
         
        self.model.compile( 
            optimizer=optimizer, 
            loss=SparseCategoricalCrossentropy(from_logits=False), 
            metrics=[SparseCategoricalAccuracy()], 
            run_eagerly=False 
        ) 
 
        print(f"Model built successfully! Parameters: {self.model.count_params():,}") 
         
        self._build_inference_models() 
        return self.model 
 
    def _build_inference_models(self): 
        print("Building inference models...") 
 
        encoder_inputs_inf = Input( 
            shape=(self.preprocessor.max_question_length,), 
            name='encoder_input_inf' 
        ) 
        encoder_embedding_layer = self.model.get_layer('encoder_embedding') 
        encoder_lstm_1_layer = self.model.get_layer('encoder_lstm_1') 
        encoder_lstm_2_layer = self.model.get_layer('encoder_lstm_2') 
 
        encoder_embedded_inf = encoder_embedding_layer(encoder_inputs_inf) 
        encoder_outputs_1_inf, fh1_inf, fc1_inf, bh1_inf, bc1_inf = encoder_lstm_1_layer(encoder_embedded_inf) 
        encoder_outputs_inf, fh2_inf, fc2_inf, bh2_inf, bc2_inf = encoder_lstm_2_layer(encoder_outputs_1_inf) 
         
        state_h_inf = Concatenate(name='encoder_inf_state_h')([fh2_inf, bh2_inf]) 
        state_c_inf = Concatenate(name='encoder_inf_state_c')([fc2_inf, bc2_inf]) 
 
        self.encoder_model = Model( 
            inputs=encoder_inputs_inf, 
            outputs=[encoder_outputs_inf, state_h_inf, state_c_inf], 
            name='encoder_model_inference' 
        ) 
 
        decoder_inputs_inf    = Input(shape=(1,), name='decoder_input_inf') 
        decoder_state_input_h = Input(shape=(self.decoder_units,), name='decoder_state_h') 
        decoder_state_input_c = Input(shape=(self.decoder_units,), name='decoder_state_c') 
        encoder_outputs_input = Input(shape=(self.preprocessor.max_question_length, self.encoder_units), 
                                      name='encoder_outputs_inf') 
 
        decoder_embedding_layer = self.model.get_layer('decoder_embedding') 
        decoder_lstm_layer      = self.model.get_layer('decoder_lstm') 
        attention_layer         = self.model.get_layer('attention') 
        decoder_ffn1_layer      = self.model.get_layer('decoder_ffn1') 
        decoder_ffn2_layer      = self.model.get_layer('decoder_ffn2') 
        output_layer            = self.model.get_layer('output_layer') 
 
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
        decoder_combined_inf = LayerNormalization(epsilon=1e-6, name='decoder_norm_inf')(decoder_combined_inf) 
         
        decoder_ffn_inf = decoder_ffn1_layer(decoder_combined_inf) 
        decoder_ffn_inf = decoder_ffn2_layer(decoder_ffn_inf) 
        decoder_combined_inf = Add(name='decoder_residual_inf')([decoder_combined_inf, decoder_ffn_inf]) 
        decoder_combined_inf = LayerNormalization(epsilon=1e-6, name='decoder_norm2_inf')(decoder_combined_inf) 
         
        decoder_outputs_final_inf = output_layer(decoder_combined_inf) 
 
        self.decoder_model = Model( 
            inputs=[ 
                decoder_inputs_inf, 
                encoder_outputs_input, 
                decoder_state_input_h, 
                decoder_state_input_c 
            ], 
            outputs=[decoder_outputs_final_inf, state_h_inf2, state_c_inf2], 
            name='decoder_model_inference' 
        ) 
 
        print("Inference models built successfully!") 
    
    def train_model(self, epochs=None, batch_size=None, use_lr_schedule=False):
        """
        Train the LSTM with optional LR scheduling.
        - epochs, batch_size: override defaults if provided
        - use_lr_schedule: if True, wraps the LR in a CosineDecayRestarts schedule
        """
        if epochs is None:
            epochs = self.epochs
        if batch_size is None:
            batch_size = self.batch_size

        print(f"Training model for {epochs} epochs, batch size {batch_size} (LR schedule = {use_lr_schedule})")

        if not hasattr(self, 'model') or self.model is None:
            self.build_model()

        if use_lr_schedule:
            steps_per_epoch = len(self.preprocessor.splits['train']['questions']) // batch_size
            lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
                initial_learning_rate=self.learning_rate,
                first_decay_steps=steps_per_epoch,
                t_mul=2.0,
                m_mul=0.9,
                alpha=0.1
            )
            optimizer = Adam(learning_rate=lr_schedule)
            reduce_lr = None  #cannot use ReduceLROnPlateau with a schedule
        else:
            optimizer = Adam(learning_rate=self.learning_rate)
            reduce_lr = ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
            )

        self.model.compile(
            optimizer=optimizer,
            loss=SparseCategoricalCrossentropy(),
            metrics=[SparseCategoricalAccuracy()]
        )

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
            ModelCheckpoint(
                filepath=f"{self.checkpoint_dir}/best_model.h5",
                monitor="val_loss", save_best_only=True, verbose=1
            ),
            CSVLogger(f"{self.logs_dir}/train_log.csv", append=True),
            TensorBoard(log_dir=self.logs_dir, histogram_freq=1)
        ]
        if reduce_lr:
            callbacks.append(reduce_lr)

        history = self.model.fit(
            [ self.preprocessor.splits["train"]["questions"],            #Input array
            self.preprocessor.splits["train"]["answer_decoder_input"] ],
            self.preprocessor.splits["train"]["answer_decoder_target"],  #Output
            validation_data=(
                [ self.preprocessor.splits["val"]["questions"],          #Validation input
                self.preprocessor.splits["val"]["answer_decoder_input"] ],
                self.preprocessor.splits["val"]["answer_decoder_target"] #Validation output
            ),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
            shuffle=True
        )

        final_path = f"{self.model_dir}/model_final.keras"
        self.model.save(final_path)
        print(f"Saved trained model to: {final_path}")

        return history

    
    def predict_answer_improved(self, question_text: str, max_length: int = None, 
                               temperature: float = 0.3, top_k: int = 50, top_p: float = 0.92) -> str: 
        if not self.model or not self.encoder_model or not self.decoder_model: 
            raise ValueError("Model not trained or inference models not built!") 
 
        if max_length is None: 
            max_length = min(self.preprocessor.max_answer_length, 80) 
 
        question_seq = self.preprocessor.tokenizer_questions.texts_to_sequences([question_text.lower().strip()]) 
        if not question_seq or not question_seq[0]: 
            return "I couldn't understand the question." 
 
        question_padded = pad_sequences( 
            question_seq, 
            maxlen=self.preprocessor.max_question_length, 
            padding='post', 
            truncating='post' 
        ) 
 
        encoder_outputs, state_h, state_c = self.encoder_model.predict(question_padded, verbose=0) 
        states_value = [state_h, state_c] 
 
        target_seq = np.zeros((1, 1), dtype='int32') 
        target_seq[0, 0] = 1 
 
        decoded_tokens = [] 
        attention_weights = [] 
 
        for step in range(max_length): 
            output_tokens, h, c = self.decoder_model.predict( 
                [target_seq, encoder_outputs, states_value[0], states_value[1]], verbose=0 
            ) 
 
            preds = output_tokens[0, 0, :].astype('float64') 
             
            if temperature == 0: 
                sampled_token_index = np.argmax(preds) 
            else: 
                preds = preds / temperature 
                 
                if top_k > 0: 
                    top_k_indices = np.argpartition(preds, -top_k)[-top_k:] 
                    filtered_preds = np.full_like(preds, -np.inf) 
                    filtered_preds[top_k_indices] = preds[top_k_indices] 
                    preds = filtered_preds 
 
                preds_exp = np.exp(preds - np.max(preds)) 
                preds_probs = preds_exp / (np.sum(preds_exp) + 1e-8) 
 
                if top_p < 1.0: 
                    sorted_indices = np.argsort(preds_probs)[::-1] 
                    cumsum_probs = np.cumsum(preds_probs[sorted_indices]) 
                    cutoff_idx = np.searchsorted(cumsum_probs, top_p) + 1 
                    cutoff_idx = max(1, min(cutoff_idx, len(sorted_indices))) 
                     
                    mask = np.zeros_like(preds_probs, dtype=bool) 
                    mask[sorted_indices[:cutoff_idx]] = True 
                    preds_probs = preds_probs * mask 
                    preds_probs = preds_probs / (np.sum(preds_probs) + 1e-8) 
 
                try: 
                    sampled_token_index = np.random.choice(len(preds_probs), p=preds_probs) 
                except: 
                    sampled_token_index = np.argmax(preds_probs) 
 
            if sampled_token_index in [0, 2]: 
                break 
 
            decoded_tokens.append(sampled_token_index) 
 
            target_seq = np.zeros((1, 1), dtype='int32') 
            target_seq[0, 0] = sampled_token_index 
            states_value = [h, c] 
 
        if decoded_tokens: 
            try: 
                reverse_word_index = { 
                    v: k for k, v in self.preprocessor.tokenizer_answers.word_index.items() 
                } 
                reverse_word_index[0] = '' 
                reverse_word_index[1] = '<START>' 
                reverse_word_index[2] = '<END>' 
 
                decoded_words = [] 
                for token_id in decoded_tokens: 
                    if token_id in reverse_word_index: 
                        word = reverse_word_index[token_id] 
                        if word and word not in ['<START>', '<END>', '<UNK>', '']: 
                            decoded_words.append(word) 
 
                if decoded_words: 
                    answer = ' '.join(decoded_words) 
                    return self._clean_generated_text(answer) 
                else: 
                    return self._fallback_answer(question_text) 
            except Exception as e: 
                print(f"Error in text conversion: {e}") 
                return self._fallback_answer(question_text) 
        else: 
            return self._fallback_answer(question_text) 
 
    def _fallback_answer(self, question_text: str) -> str: 
        fallback_responses = { 
            "what": "This refers to a specific concept or definition in cybersecurity.", 
            "how": "This involves a process or methodology in cybersecurity practices.", 
            "why": "This is important for maintaining security and protecting systems.", 
            "when": "This depends on the specific security context and requirements.", 
            "where": "This applies to various levels of network and system security." 
        } 
         
        question_lower = question_text.lower() 
        for keyword, response in fallback_responses.items(): 
            if question_lower.startswith(keyword): 
                return response 
         
        return "This is a cybersecurity-related topic that requires specific domain knowledge." 
 
    def _clean_generated_text(self, text: str) -> str: 
        if not text: 
            return "No answer available." 
             
        words = text.split() 
        cleaned_words = [] 
         
        for word in words: 
            if len(word) > 1 and word.isalnum(): 
                cleaned_words.append(word) 
            elif word in ['.', '!', '?', ',', ';', ':']: 
                cleaned_words.append(word) 
         
        if not cleaned_words: 
            return "No clear answer could be generated." 
             
        text = ' '.join(cleaned_words) 
         
        if text: 
            text = text[0].upper() + text[1:] if len(text) > 1 else text.upper() 
         
        if text and not text[-1] in '.!?': 
            text += '.' 
             
        return text 
 
    def evaluate_model(self): 
        print("Evaluating model on test set...") 
        test_loss, test_acc = self.model.evaluate( 
            [self.preprocessor.splits['test']['questions'], 
            self.preprocessor.splits['test']['answer_decoder_input']], 
            self.preprocessor.splits['test']['answer_decoder_target'], 
            batch_size=self.batch_size, 
            verbose=1 
        ) 
        print(f"Test Loss: {test_loss:.4f}") 
        print(f"Test Accuracy: {test_acc:.4f}") 
 
        print("\n--> Sample Predictions <--") 
        sample_questions = [ 
            "What is cybersecurity?", 
            "How does encryption work?",  
            "What is a firewall?", 
            "Explain malware detection", 
            "What is network security?" 
        ] 
        for question in sample_questions: 
            answer = self.predict_answer_improved(question, max_length=60, temperature=0.2) 
            print(f"Q: {question}") 
            print(f"A: {answer}") 
            print("\n") 
 
    def save_model(self, model_path=None): 
        """
        Save the current LSTM model. 
        """
        if model_path is None:
            
            model_path = f"{self.model_dir}/model_final.keras"

        try:
            self.model.save(model_path)
            print(f"Model successfully saved to: {model_path}")
            return True
        except Exception as e:
            print(f"[Save ERROR] {e}")
            return False
 
    def load_model(self, model_path: str = None) -> bool:
        if model_path is None:
            model_path = os.path.join(self.model_dir, "model_final.keras")

        if not os.path.exists(model_path):
            print(f"[Load ERROR] Model file not found: {model_path}")
            return False

        try:
            from tensorflow.keras.models import load_model as keras_load_model
            
            self.model = keras_load_model(model_path)
            print(f"[Load OK] Model successfully loaded from: {model_path}")

            self.encoder_units   = 256
            self.decoder_units   = 256
            self.attention_units = 128
            
            self._build_inference_models()

            return True
        except Exception as e:
            print(f"[Load ERROR] Failed to load model: {e}")
            return False
        
    def fine_tune_on_feedback(self, feedback_list: list, epochs: int = 10, batch_size: int = 4): 
        if not feedback_list: 
            print("No feedback to train on.") 
            return 
 
        print(f"Preparing to fine-tune on {len(feedback_list)} feedback examples...") 
 
        q_seqs = [] 
        a_seqs = [] 
        for entry in feedback_list: 
            q     = entry["question"].lower().strip() 
            a     = entry["llm_reference_answer"].lower().strip() 
            q_seq = self.preprocessor.tokenizer_questions.texts_to_sequences([q])[0] 
            a_seq = self.preprocessor.tokenizer_answers.texts_to_sequences([a])[0] 
            if q_seq and a_seq: 
                q_seqs.append(q_seq) 
                a_seqs.append(a_seq) 
        
 
        if not q_seqs: 
            print("No valid feedback pairs after tokenization. Aborting fine-tuning.") 
            return 
 
        Qp = pad_sequences(q_seqs, maxlen=self.preprocessor.max_question_length, padding="post",truncating="post") 
        Ap = pad_sequences(a_seqs, maxlen=self.preprocessor.max_answer_length, padding="post",truncating="post") 
 
        def build_decoder_sequences(all_seqs: np.ndarray, max_len: int): 
            N = all_seqs.shape[0] 
            dec_input = np.zeros((N, max_len), dtype=np.int32) 
            dec_target = np.zeros((N, max_len), dtype=np.int32) 
 
            for i, seq in enumerate(all_seqs): 
                actual_len = int(np.sum(seq > 0)) 
                capped_len = min(actual_len, max_len - 1) 
 
                dec_input[i, 0] = 1   
                if capped_len > 0: 
                    dec_input[i, 1:capped_len+1] = seq[:capped_len] 
                    dec_target[i, :capped_len] = seq[:capped_len] 
 
                if capped_len < max_len: 
                    dec_target[i, capped_len] = 2   
 
            return dec_input, dec_target 
 
        ans_dec_inp, ans_dec_tgt = build_decoder_sequences(Ap, self.preprocessor.max_answer_length) 
 
        callbacks = [ 
            ReduceLROnPlateau( 
                monitor='val_loss', 
                factor=0.7, 
                patience=3, 
                verbose=1, 
                min_delta=1e-4, 
                cooldown=2, 
                min_lr=1e-6 
            ), 
            EarlyStopping( 
                monitor='val_loss', 
                patience=5, 
                verbose=1, 
                restore_best_weights=True 
            ) 
        ] 
 
        print(f"Starting fine-tuning for {epochs} epochs...") 
 
        self.model.fit( 
            [Qp, ans_dec_inp], 
            ans_dec_tgt, 
            epochs=epochs, 
            batch_size=batch_size, 
            verbose=1, 
            shuffle=True, 
            validation_split=0.1, 
            callbacks=callbacks 
        ) 
 
        self.save_model() 
        print("Fine-tuning completed and model saved.") 
  
class QAApp: 
    """ 
    Main application class that provides a menu: 
      1) Train new model 
      2) Load existing model 
      3) Interactive QA session 
      4) Test feedback (fine-tune on collected feedback) 
      5) Exit 
    It uses ImprovedLSTMModel and CybersecurityLLM under the hood. 
    """ 
    def __init__(self, preprocessor_path: str): 
         
        self.preprocessor = LSTMQAPreprocessorLoader(preprocessor_path) 
  
        llm_client = None 
        try: 
            groq_api_key = os.getenv('GROQ_API_KEY') or "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" 
            if groq_api_key: 
                llm_client = CybersecurityLLM(api_key=groq_api_key) 
                print("LLM client initialized successfully!") 
            else: 
                print("No GROQ_API_KEY found, LLM fallback disabled.") 
        except Exception as e: 
            print(f"LLM initialization failed: {e}") 
            llm_client = None 

        self.model = ImprovedLSTMModel( 
            preprocessor=self.preprocessor, 
            llm_client=llm_client, 
            model_name="cybersec_lstm_v2" 
        ) 
 
        self.feedback_pool = [] 
 
    def menu(self): 
        """ 
        Display menu and handle user choices. 
        """ 
        while True: 
            print("\n") 
            print(" QA_LSTM + LLM APPLICATION MENU ") 
            print("\n") 
            print("1) Train new LSTM model") 
            print("2) Load existing trained model") 
            print("3) Interactive Q&A session") 
            print("4) Test feedback (fine-tune on collected feedback)") 
            print("5) Exit") 
            choice = input("Choose an option (1-5): ").strip() 
 
            if choice == "1": 
                self.train_new_model() 
            elif choice == "2": 
                self.load_existing_model() 
            elif choice == "3": 
                self.interactive_qa_session() 
            elif choice == "4": 
                if self.model is None: 
                    print("No trained model loaded or trained. Please train or load a model first.") 
                    return 
                self.test_feedback() 
            elif choice == "5": 
                print("Exiting application. Goodbye!") 
                break 
            else: 
                print("Invalid choice. Please enter a number between 1 and 5.") 
 
    def train_new_model(self): 
        """ 
        Train a brand-new LSTM model from scratch using the preprocessed data. 
        """ 
         
        self.model.embedding_dim = 300 
        self.model.encoder_units = 256 
        self.model.decoder_units = 256 
        self.model.attention_units = 128 
        self.model.dropout_rate = 0.3 
        self.model.learning_rate = 0.001 
        self.model.batch_size = 16 
        self.model.epochs = 30 
         
        print("\n---> TRAIN NEW LSTM MODEL <---") 
        self.model.build_model() 
        self.model.model.summary() 
        self.model.train_model(epochs=30, batch_size=16, use_lr_schedule=True) 
 
    def load_existing_model(self): 
        """ 
        Load an existing trained model from disk. 
        """ 
        print("\n--> LOADED EXISTING MODEL <--") 
        success = self.model.load_model() 
        if not success: 
            print("Failed to load model.") 
 
    def interactive_qa_session(self): 
        """ 
        Run an interactive Q&A session where the user asks questions, 
        receives both LSTM and LLM answers, and provides feedback. 
        Incorrect LSTM answers (based on user remark) cause LLM answer 
        to be stored for later fine-tuning. 
        """ 
        if not self.model.model or not self.model.encoder_model or not self.model.decoder_model: 
            print("No trained model loaded or trained. Please train or load a model first.") 
            return 
 
        if not self.model.llm_client: 
            print("LLM client is not available. Interactive QA will only use LSTM.") 
        print("\n<== INTERACTIVE Q&A SESSION ==>") 
        print("Type 'quit' or 'exit' to return to the menu.\n") 
 
        while True: 
            question = input("Enter your question: ").strip() 
            if question.lower() == "quit" or question.lower() == "exit": 
                break 
            if not question: 
                print("Please type a non-empty question.") 
                continue 
             
            print("\n[LSTM is thinking...]") 
            t0 = datetime.now() 
            lstm_ans = self.model.predict_answer_improved(question, temperature=0.5) 
            dt_lstm = (datetime.now() - t0).total_seconds() 
            print(f"LSTM Answer (in {dt_lstm:.2f}s):\n\n{lstm_ans}") 
 
            if self.model.llm_client: 
                print("\n[LLM is thinking...]") 
                t1 = datetime.now() 
                if self.model.llm_client.is_cybersecurity_related(question): 
                    llm_ans = self.model.llm_client.generate_answer(question) 
                else: 
                    llm_ans = "This question isn't cybersecurity-related. I specialize in cybersecurity topics." 
                dt_llm = (datetime.now() - t1).total_seconds() 
                print(f"\nLLM Answer (in {dt_llm:.2f}s):\n\n{llm_ans}") 
            else: 
                llm_ans = "" 
                dt_llm = 0.0 
 
            print("\n--> LSTM Feedback <--") 
            rst = input("Is the LSTM answer correct? (yes/no/partial): ").strip().lower() 
            rating = input("Rate LSTM answer quality (1-5): ").strip() #Not used currently, but can be extended
            remark = input("Specific feedback on LSTM answer: ").strip() 
 
            if rst in ("no", "incorrect", "wrong","n"): 
                if llm_ans: 
                    self.feedback_pool.append({ 
                        "timestamp": datetime.now().isoformat(), 
                        "question": question, 
                        "lstm_answer": lstm_ans, 
                        "llm_reference_answer": llm_ans, 
                        "user_remark": remark, 
                        "lstm_rating": rating 
                    }) 
                     
                    print("Stored LLM answer for retraining later.") 
                else: 
                    print("No LLM answer available to store as reference.") 
            
            print("\nFeedback recorded. You may ask another question or type 'quit' to return to the menu.") 
 
        print(f"\nSession ended. You accumulated {len(self.feedback_pool)} feedback items.") 
 
    def test_feedback(self): 
        """ 
        Fine-tune the model on the collected feedback (LLM-provided answers). 
        This will use the feedback_pool to perform a short fine-tuning pass. 
        """ 
        print("\n--> TEST FEEDBACK (FINE-TUNE) <--") 
        if not self.feedback_pool: 
            print("No feedback collected yet. Run an interactive QA session first.") 
            return 
          
        if not self.model.model or not self.model.encoder_model or not self.model.decoder_model: 
            print("No trained model loaded or trained. Please train or load a model first.") 
            return 
 
        self.model.fine_tune_on_feedback(self.feedback_pool, epochs=8, batch_size=16) 
 
        self.feedback_pool = [] 
        print("Feedback pool cleared after fine-tuning.") 
 
if __name__ == "__main__": 
    preprocessor_path = "./preprocessed_data" 
    if not os.path.exists(preprocessor_path): 
        print(f"Preprocessor path '{preprocessor_path}' not found. Please prepare the preprocessed data first.") 
        exit(1) 
     
    app = QAApp(preprocessor_path=preprocessor_path) 
    app.menu() 


 