# import os
# import json
# import numpy as np
# import pandas as pd
# from datetime import datetime
# import tensorflow as tf
# from tensorflow.keras.models import Model, load_model
# from tensorflow.keras.layers import (
#     Input, LSTM, Dense, Embedding, Dropout, BatchNormalization,
#     Attention, Add, LayerNormalization, Concatenate, Bidirectional,
#     MultiHeadAttention, GlobalAveragePooling1D
# )
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import (
#     EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,
#     CSVLogger, TensorBoard
# )
# from tensorflow.keras.losses import SparseCategoricalCrossentropy
# from tensorflow.keras.metrics import SparseCategoricalAccuracy
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.regularizers import l2
# import pickle
# import warnings
# import requests
# import time
# import re
# from groq import Groq
# warnings.filterwarnings('ignore')

# # Set mixed precision for better performance
# tf.keras.mixed_precision.set_global_policy('mixed_float16')

# class CybersecurityLLM:
#     """
#     Cybersecurity-focused LLM integration using Groq
#     """
#     def __init__(self, api_key=None, model_name="llama3-8b-8192"):
#         """
#         Initialize the LLM client
        
#         Args:
#             api_key: Groq API key 
#             model_name: Model to use (default: llama3-8b-8192)
#         """
#         self.api_key = api_key or os.getenv('GROQ_API_KEY')
#         if not self.api_key:
#             raise ValueError("Groq API key is required. Set GROQ_API_KEY environment variable or pass it directly.")
        
#         self.client = Groq(api_key=self.api_key)
#         self.model_name = model_name
#         self.max_tokens = 150 

#         self.system_prompt = """You are a cybersecurity expert AI assistant specializing in:
# - Cybersecurity fundamentals and best practices
# - Data privacy and protection regulations
# - Malware analysis and threat detection
# - Computer networking and network security
# - Operating system security
# - Artificial intelligence security implications
# - Digital forensics techniques
# - Steganography and cryptography
# - Penetration testing methodologies
# - Incident response and threat hunting

# Provide accurate, concise, and educational responses. Keep answers under 150 words and format them clearly for educational purposes. Focus on practical knowledge that would be valuable for cybersecurity learning."""

#         self._compile_keyword_pattern()
    
#     def _compile_keyword_pattern(self):
#         """
#         Create a single regex pattern from all keywords, matching whole words
#         (case-insensitive). This is faster than running `any(keyword in question)`
#         repeatedly, and avoids partial-substring mistakes.
#         """
#         cybersec_keywords = [
#             r'\bcybersecurity\b', r'\bsecurity\b', r'\binformation security\b', r'\binfosec\b',
#             r'\bdata protection\b', r'\bdata privacy\b', r'\bprivacy\b',
#             r'\bcompliance\b', r'\brisk management\b',
#             r'\bmalware\b', r'\bvirus\b', r'\btrojan\b', r'\bransomware\b', r'\bspyware\b',
#             r'\brootkit\b', r'\bworm\b', r'\badware\b', r'\bbackdoor\b', r'\bkeylogger\b',
#             r'\bzero[\s\-]?day\b', r'\bapt\b', r'\bbotnet\b',
#             r'\bthreat\b', r'\bthreats\b', r'\bcyber threat\b', r'\bcyberthreat\b',
#             r'\bfirewall\b', r'\bids\b', r'\bips\b', r'\bvpn\b', r'\bproxy\b',
#             r'\bport scanning\b', r'\bnetwork sniffing\b', r'\bintrusion detection\b',
#             r'\bintrusion prevention\b', r'\bpacket inspection\b', r'\bnetwork segmentation\b',
#             r'\bxss\b', r'\bsql injection\b', r'\bcsrf\b', r'\bclickjacking\b',
#             r'\bdirectory traversal\b', r'\bsession hijacking\b', r'\btoken manipulation\b',
#             r'\binput validation\b', r'\brate limiting\b',
#             r'\bauthentication\b', r'\bauthorization\b', r'\bmfa\b', r'\b2fa\b',
#             r'\bsso\b', r'\boauth\b', r'\bsaml\b', r'\baccess control\b',
#             r'\bleast privilege\b', r'\bidentity management\b',
#             r'\bencryption\b', r'\bdecryption\b', r'\bhashing\b', r'\bssl\b', r'\btls\b',
#             r'\bcertificate\b', r'\bca\b', r'\bpk\b', r'\brsa\b', r'\baes\b',
#             r'\bkey exchange\b', r'\bpublic key\b', r'\bprivate key\b', r'\bdigital signature\b',
#             r'\bcloud security\b', r'\bcontainer security\b', r'\bkubernetes security\b',
#             r'\bsecrets management\b', r'\binfrastructure as code\b', r'\bdevsecops\b',
#             r'\bshift left\b', r'\biam\b', r'\bcloud misconfigurations\b',
#             r'\bthreat intelligence\b', r'\bthreat hunting\b', r'\bioc\b', r'\btactics\b',
#             r'\bmitre att&ck\b', r'\bincident response\b', r'\bforensics\b', r'\bsiem\b',
#             r'\bedr\b', r'\bsoar\b', r'\blog analysis\b', r'\bcorrelation\b',
#             r'\bphishing\b', r'\bsmishing\b', r'\bvishing\b', r'\bsocial engineering\b',
#             r'\bbrute force\b', r'\bcredential stuffing\b', r'\bddos\b', r'\bdos\b',
#             r'\bwatering hole\b', r'\bman in the middle\b', r'\beavesdropping\b',
#             r'\bgdpr\b', r'\bccpa\b', r'\bhipaa\b', r'\bpci dss\b', r'\biso 27001\b',
#             r'\bnist\b', r'\bndpr\b', r'\bcisa\b', r'\bcmmc\b', r'\bsoc 2\b',
#             r'\bai security\b', r'\bmachine learning security\b', r'\bblockchain security\b',
#             r'\bquantum encryption\b', r'\bpost-quantum cryptography\b', r'\bdeepfake detection\b',
#             r'\bbug bounty\b', r'\bpenetration testing\b', r'\bpentest\b', r'\bred teaming\b',
#             r'\bblue team\b', r'\bpurple team\b', r'\bsecurity awareness\b', r'\bsecurity audit\b',
#             r'\bzero trust\b', r'\bsandboxing\b', r'\bhoneypot\b', r'\bvulnerability\b',
#             r'\bexploit\b', r'\bpatching\b', r'\bmonitoring\b', r'\bendpoint security\b',
#             r'\bantivirus\b', r'\bsecurity operations center\b', r'\bincident playbook\b'
#         ]

#         pattern = r"(?i)(" + "|".join(cybersec_keywords) + r")"
#         self._keyword_regex = re.compile(pattern)    

#     def generate_answer(self, question, context=None):
#         """Generate answer using Groq LLM"""
#         try:
#             user_prompt = f"Question: {question}"
#             if context:
#                 user_prompt = f"Context: {context}\n\n{user_prompt}"
            
#             chat_completion = self.client.chat.completions.create(
#                 messages=[
#                     {"role": "system", "content": self.system_prompt},
#                     {"role": "user", "content": user_prompt}
#                 ],
#                 model=self.model_name,
#                 max_tokens=self.max_tokens,
#                 temperature=0.7,
#                 top_p=0.9,
#                 stream=False,
#             )
            
#             answer = chat_completion.choices[0].message.content.strip()
#             return self.format_answer(answer)
            
#         except Exception as e:
#             print(f"Error generating LLM answer: {e}")
#             return f"Sorry, I couldn't generate an answer due to an error: {str(e)}"
    
#     def format_answer(self, answer):
#         """Format the answer for better readability"""
#         lines = answer.split('\n')
#         formatted_lines = []
        
#         for line in lines:
#             line = line.strip()
#             if line:
#                 if line.endswith(':'):
#                     formatted_lines.append(line)
#                 elif line.startswith('-') or line.startswith('•'):
#                     formatted_lines.append(f"  {line}")
#                 else:
#                     formatted_lines.append(line)
        
#         formatted_answer = '\n'.join(formatted_lines)
#         words = formatted_answer.split()
#         if len(words) > 140:  
#             formatted_answer = ' '.join(words[:140]) + "..."
        
#         return formatted_answer
    
#     def is_cybersecurity_related(self, question: str) -> bool:
#         """Check if a given question is cybersecurity-related"""
#         if not question or not question.strip():
#             return False
#         found = self._keyword_regex.search(question)
#         return bool(found)

# class ImprovedLSTMQAPreprocessorLoader:
#     """Enhanced preprocessor loader with better data handling"""
#     def __init__(self, preprocessor_path):
#         self.preprocessor_path = preprocessor_path
#         self.has_remarks = False
#         self.has_improvements = False
#         self.load_preprocessor_data()

#     def load_preprocessor_data(self):
#         """Load all preprocessor components with improved error handling"""
#         print("Loading preprocessor data...")
        
#         # Load configuration
#         with open(f"{self.preprocessor_path}/config.json", 'r') as f:
#             self.config = json.load(f)
            
#         # Load tokenizers
#         with open(f"{self.preprocessor_path}/tokenizer_questions.pkl", 'rb') as f:
#             self.tokenizer_questions = pickle.load(f)
#         with open(f"{self.preprocessor_path}/tokenizer_answers.pkl", 'rb') as f:
#             self.tokenizer_answers = pickle.load(f)
            
#         # Load preprocessed sequences
#         self.X_questions = np.load(f"{self.preprocessor_path}/sequences.npz")['questions']
#         self.X_answers = np.load(f"{self.preprocessor_path}/sequences.npz")['answers']
        
#         # Handle optional data
#         try:
#             sequences_data = np.load(f"{self.preprocessor_path}/sequences.npz")
#             if 'remarks' in sequences_data:
#                 self.X_remarks = sequences_data['remarks']
#                 if np.any(self.X_remarks > 2):
#                     self.has_remarks = True
#                     with open(f"{self.preprocessor_path}/tokenizer_remarks.pkl", 'rb') as f:
#                         self.tokenizer_remarks = pickle.load(f)
#                 else:
#                     self.has_remarks = False
#                     self.tokenizer_remarks = self.tokenizer_answers
#             else:
#                 self.has_remarks = False
#                 self.tokenizer_remarks = self.tokenizer_answers
#                 self.X_remarks = np.zeros_like(self.X_answers)
#         except:
#             self.has_remarks = False
#             self.tokenizer_remarks = self.tokenizer_answers
#             self.X_remarks = np.zeros_like(self.X_answers)

#         # Load splits
#         try:
#             with open(f"{self.preprocessor_path}/splits.pkl", 'rb') as f:
#                 self.splits_indices = pickle.load(f)
#         except:
#             self.splits_indices = {}

#         # Set sequence lengths
#         self.max_question_length = self.config.get('max_question_length', self.X_questions.shape[1])
#         self.max_answer_length = self.config.get('max_answer_length', self.X_answers.shape[1])
#         self.max_remark_length = self.config.get('max_remark_length', self.max_answer_length)

#         # Set vocabulary sizes
#         self.vocab_size_questions = len(self.tokenizer_questions.word_index) + 1
#         self.vocab_size_answers = len(self.tokenizer_answers.word_index) + 1
#         self.vocab_size_remarks = len(self.tokenizer_remarks.word_index) + 1

#         print(f"Preprocessor data loaded successfully!")
#         print(f"Questions vocab: {self.vocab_size_questions}, Answers vocab: {self.vocab_size_answers}")
#         print(f"Max lengths - Questions: {self.max_question_length}, Answers: {self.max_answer_length}")
        
#         self.prepare_training_data()

#     def prepare_training_data(self):
#         """Prepare training data splits with improved decoder setup"""
#         print("Preparing training data...")
        
#         if not self.splits_indices:
#             n_samples = len(self.X_questions)
#             indices = np.arange(n_samples)
#             np.random.shuffle(indices)
            
#             train_size = int(0.8 * n_samples)
#             val_size = int(0.1 * n_samples)
            
#             train_idx = indices[:train_size]
#             val_idx = indices[train_size:train_size + val_size]
#             test_idx = indices[train_size + val_size:]
            
#             self.splits_indices = {
#                 'train': train_idx,
#                 'val': val_idx,
#                 'test': test_idx
#             }

#         train_idx = self.splits_indices['train']
#         val_idx = self.splits_indices['val']
#         test_idx = self.splits_indices['test']
        
#         print(f"Dataset split sizes:")
#         print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

#         def prepare_decoder_data_improved(sequences, max_length):
#             """Improved decoder data preparation"""
#             # Ensure sequences are properly padded
#             if sequences.shape[1] != max_length:
#                 sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
            
#             decoder_input = np.zeros((len(sequences), max_length), dtype=np.int32)
#             decoder_target = np.zeros((len(sequences), max_length), dtype=np.int32)
            
#             for i, seq in enumerate(sequences):
#                 # Find actual sequence length (excluding padding)
#                 non_zero_indices = np.where(seq > 0)[0]
#                 if len(non_zero_indices) > 0:
#                     actual_length = non_zero_indices[-1] + 1
#                 else:
#                     actual_length = 0
                
#                 # Prepare decoder input (start with <START> token = 1)
#                 decoder_input[i, 0] = 1  # <START> token
#                 if actual_length > 0:
#                     copy_length = min(actual_length, max_length - 1)
#                     decoder_input[i, 1:copy_length + 1] = seq[:copy_length]
                
#                 # Prepare decoder target (shifted by one position)
#                 if actual_length > 0:
#                     copy_length = min(actual_length, max_length)
#                     decoder_target[i, :copy_length] = seq[:copy_length]
#                     # Add <END> token if there's space
#                     if copy_length < max_length:
#                         decoder_target[i, copy_length] = 2  # <END> token
                        
#             return decoder_input, decoder_target

#         # Prepare decoder data
#         answer_decoder_input, answer_decoder_target = prepare_decoder_data_improved(
#             self.X_answers, self.max_answer_length
#         )
        
#         # Store splits
#         self.splits = {
#             'train': {
#                 'questions': self.X_questions[train_idx],
#                 'answers': self.X_answers[train_idx],
#                 'answer_decoder_input': answer_decoder_input[train_idx],
#                 'answer_decoder_target': answer_decoder_target[train_idx],
#             },
#             'val': {
#                 'questions': self.X_questions[val_idx],
#                 'answers': self.X_answers[val_idx],
#                 'answer_decoder_input': answer_decoder_input[val_idx],
#                 'answer_decoder_target': answer_decoder_target[val_idx],
#             },
#             'test': {
#                 'questions': self.X_questions[test_idx],
#                 'answers': self.X_answers[test_idx],
#                 'answer_decoder_input': answer_decoder_input[test_idx],
#                 'answer_decoder_target': answer_decoder_target[test_idx],
#             }
#         }
        
#         print("Training data prepared successfully!")

# class ImprovedLSTMModel:
#     """Significantly improved LSTM model for better text generation"""
    
#     def __init__(self, preprocessor, llm_client=None, model_name="improved_lstm_qa"):
#         self.preprocessor = preprocessor
#         self.llm_client = llm_client
#         self.model_name = model_name
#         self.model = None
#         self.encoder_model = None
#         self.decoder_model = None
        
#         # Improved hyperparameters
#         self.embedding_dim = 256
#         self.encoder_units = 512
#         self.decoder_units = 512
#         self.attention_units = 256
#         self.dropout_rate = 0.2
#         self.learning_rate = 0.0005
#         self.batch_size = 32
#         self.epochs = 50
        
#         # Model directories
#         self.model_dir = f"./models/{self.model_name}"
#         self.checkpoint_dir = f"{self.model_dir}/checkpoints"
#         self.logs_dir = f"{self.model_dir}/logs"
        
#         os.makedirs(self.model_dir, exist_ok=True)
#         os.makedirs(self.checkpoint_dir, exist_ok=True)
#         os.makedirs(self.logs_dir, exist_ok=True)
        
#         print(f"Initialized Improved LSTM Model: {self.model_name}")

#     def build_model(self):
#         """
#         Build improved sequence-to-sequence model with attention for training.
#         """
#         print("Building improved seq2seq model with attention...")

#         # ----- ENCODER (training) -----
#         # 1) Encoder input: a full question (integer‐encoded, padded to max_question_length)
#         encoder_inputs = Input(
#             shape=(self.preprocessor.max_question_length,),
#             name='encoder_input'
#         )
#         # 2) Shared embedding layer for the encoder
#         encoder_embedding = Embedding(
#             input_dim=self.preprocessor.vocab_size_questions,
#             output_dim=self.embedding_dim,
#             mask_zero=True,
#             name='encoder_embedding'
#         )(encoder_inputs)

#         # 3) Bidirectional LSTM encoder
#         #    - return_sequences=True so we get the full sequence of encoder outputs
#         #    - return_state=True so we get both forward and backward states
#         encoder_lstm = Bidirectional(
#             LSTM(
#                 self.encoder_units,
#                 return_sequences=True,
#                 return_state=True,
#                 dropout=self.dropout_rate,
#                 recurrent_dropout=self.dropout_rate / 2
#             ),
#             name='encoder_lstm'
#         )
#         # When you call a Bidirectional(LSTM(..., return_state=True)) on encoder_embedding,
#         # you get a tuple:
#         #   (encoder_seq_outputs,
#         #    forward_h, forward_c,
#         #    backward_h, backward_c)
#         encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder_lstm(encoder_embedding)

#         # 4) Concatenate the forward/backward final states into single vectors
#         state_h = Concatenate(name='encoder_state_h')([forward_h, backward_h])
#         state_c = Concatenate(name='encoder_state_c')([forward_c, backward_c])
#         encoder_states = [state_h, state_c]  # used to initialize the decoder

#         # ----- DECODER (training) -----
#         # Decoder input: a full answer sequence (integer‐encoded, padded to max_answer_length)
#         decoder_inputs = Input(
#             shape=(self.preprocessor.max_answer_length,),
#             name='decoder_input'
#         )
#         # Shared embedding layer for the decoder
#         decoder_embedding = Embedding(
#             input_dim=self.preprocessor.vocab_size_answers,
#             output_dim=self.embedding_dim,
#             mask_zero=True,
#             name='decoder_embedding'
#         )(decoder_inputs)

#         # Decoder LSTM: note we double decoder_units because the encoder’s forward+backward states get concatenated
#         decoder_lstm = LSTM(
#             self.decoder_units * 2,
#             return_sequences=True,
#             return_state=True,
#             dropout=self.dropout_rate,
#             recurrent_dropout=self.dropout_rate / 2,
#             name='decoder_lstm'
#         )
#         # Passing initial_state=encoder_states:
#         decoder_outputs, _, _ = decoder_lstm(
#             decoder_embedding,
#             initial_state=encoder_states
#         )

#         # ----- ATTENTION (training) -----
#         # We use a MultiHeadAttention layer that has already been configured/trained
#         attention = MultiHeadAttention(
#             num_heads=8,
#             key_dim=self.attention_units // 8,
#             name='attention'
#         )
#         attention_output = attention(
#             query=decoder_outputs,
#             value=encoder_outputs,
#             key=encoder_outputs
#         )

#         # Residual + layer norm
#         decoder_combined = Add(name='decoder_attention_add')([decoder_outputs, attention_output])
#         decoder_combined = LayerNormalization(name='decoder_norm')(decoder_combined)
#         decoder_combined = Add(name='decoder_residual')([decoder_combined, decoder_outputs])

#         # ----- DECODER DENSE BLOCKS (training) -----
#         decoder_dense1 = Dense(
#             self.decoder_units,
#             activation='relu',
#             kernel_regularizer=l2(0.001),
#             name='decoder_dense1'
#         )(decoder_combined)
#         decoder_dropout = Dropout(self.dropout_rate, name='decoder_dropout')(decoder_dense1)
#         decoder_dense2 = Dense(
#             self.decoder_units // 2,
#             activation='relu',
#             kernel_regularizer=l2(0.001),
#             name='decoder_dense2'
#         )(decoder_dropout)

#         # Final softmax layer (vocab_size_answers)
#         decoder_outputs_final = Dense(
#             self.preprocessor.vocab_size_answers,
#             activation='softmax',
#             dtype='float32',
#             name='output_layer'
#         )(decoder_dense2)

#         # Build the full training model
#         self.model = Model(
#             inputs=[encoder_inputs, decoder_inputs],
#             outputs=decoder_outputs_final,
#             name='seq2seq_model'
#         )

#         # Compile with an Adam optimizer and sparse‐categorical‐crossentropy loss
#         optimizer = Adam(
#             learning_rate=self.learning_rate,
#             clipnorm=1.0,
#             beta_1=0.9,
#             beta_2=0.999,
#             epsilon=1e-7
#         )
#         self.model.compile(
#             optimizer=optimizer,
#             loss=SparseCategoricalCrossentropy(from_logits=False),
#             metrics=[SparseCategoricalAccuracy()],
#             run_eagerly=False
#         )

#         print(f"Model built successfully! Parameters: {self.model.count_params():,}")

#         # Now build the separate inference models (encoder-only and decoder-only)
#         self._build_inference_models()
#         return self.model

#     def _build_inference_models(self):
        # """
        # Build separate encoder and decoder models for inference,
        # automatically inferring the correct encoder‐output dimension.
        # """
        # print("Building inference models...")

        # # ----- 1) ENCODER INFERENCE MODEL -----
        # # Create a brand‐new Input for inference: a full question of length max_question_length
        # encoder_inputs_inf = Input(
        #     shape=(self.preprocessor.max_question_length,),
        #     name='encoder_input_inf'
        # )

        # # Grab the trained embedding and Bidirectional LSTM layers from the full model
        # encoder_embedding_layer = self.model.get_layer('encoder_embedding')
        # encoder_lstm_layer      = self.model.get_layer('encoder_lstm')

        # # Re‐apply them to the new Input
        # encoder_embedded_inf = encoder_embedding_layer(encoder_inputs_inf)
        # encoder_outputs_inf, forward_h_inf, forward_c_inf, backward_h_inf, backward_c_inf = encoder_lstm_layer(
        #     encoder_embedded_inf
        # )

        # # Concatenate forward/backward states
        # state_h_inf = Concatenate(name='encoder_inf_state_h')([forward_h_inf, backward_h_inf])
        # state_c_inf = Concatenate(name='encoder_inf_state_c')([forward_c_inf, backward_c_inf])

        # # Build the encoder‐only inference model
        # # It will output: [encoder_outputs_seq, state_h, state_c]
        # self.encoder_model = Model(
        #     inputs=encoder_inputs_inf,
        #     outputs=[encoder_outputs_inf, state_h_inf, state_c_inf],
        #     name='encoder_model_inference'
        # )

        # # ----- 2) DECODER INFERENCE MODEL -----
        # # We need to know the true shape (seq_len, feature_dim) of encoder_outputs_inf.
        # # Inspect the symbolic shape from the trained encoder_lstm_layer:
        # enc_out_shape = encoder_lstm_layer.output.shape  # TensorShape([None, seq_len, feature_dim])
        # _, seq_len, feat_dim = enc_out_shape.as_list()

        # # Create placeholders for the four inputs to the decoder at inference time:
        # #  (a) one previous token (shape=(None, 1))
        # #  (b) the previous hidden state h (shape=(None, feature_dim))
        # #  (c) the previous cell state   c (shape=(None, feature_dim))
        # #  (d) the entire encoder output sequence (shape=(None, seq_len, feature_dim))
        # decoder_inputs_inf      = Input(shape=(1,), name='decoder_input_inf')
        # decoder_state_input_h   = Input(shape=(feat_dim,), name='decoder_state_h')
        # decoder_state_input_c   = Input(shape=(feat_dim,), name='decoder_state_c')
        # encoder_outputs_input   = Input(shape=(seq_len, feat_dim), name='encoder_outputs')

        # decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

        # # Re‐use the trained decoder_embedding and decoder_lstm layers
        # decoder_embedding_layer = self.model.get_layer('decoder_embedding')
        # decoder_embedded_inf    = decoder_embedding_layer(decoder_inputs_inf)

        # decoder_lstm_layer = self.model.get_layer('decoder_lstm')
        # decoder_outputs_inf, state_h_inf2, state_c_inf2 = decoder_lstm_layer(
        #     decoder_embedded_inf,
        #     initial_state=decoder_states_inputs
        # )

        # decoder_states_inf = [state_h_inf2, state_c_inf2]

        # # Re‐use the trained attention layer
        # attention_layer = self.model.get_layer('attention')
        # attention_output_inf = attention_layer(
        #     query=decoder_outputs_inf,
        #     value=encoder_outputs_input,
        #     key=encoder_outputs_input
        # )

        # # Reconstruct the same Add + LayerNorm + Add residual logic
        # decoder_combined_inf = Add(name='decoder_attention_add_inf')([decoder_outputs_inf, attention_output_inf])
        # decoder_combined_inf = LayerNormalization(name='decoder_norm_inf')(decoder_combined_inf)
        # decoder_combined_inf = Add(name='decoder_residual_inf')([decoder_combined_inf, decoder_outputs_inf])

        # # Re‐use the trained dense/dropout layers
        # decoder_dense1_layer   = self.model.get_layer('decoder_dense1')
        # decoder_dropout_layer  = self.model.get_layer('decoder_dropout')
        # decoder_dense2_layer   = self.model.get_layer('decoder_dense2')
        # output_layer           = self.model.get_layer('output_layer')

        # decoder_dense1_inf     = decoder_dense1_layer(decoder_combined_inf)
        # decoder_dropout_inf    = decoder_dropout_layer(decoder_dense1_inf)
        # decoder_dense2_inf     = decoder_dense2_layer(decoder_dropout_inf)
        # decoder_outputs_final_inf = output_layer(decoder_dense2_inf)

        # # Build the decoder‐only inference model
        # # Inputs:  [decoder_inputs_inf, encoder_outputs_input, decoder_state_input_h, decoder_state_input_c]
        # # Outputs: [next_token_probs, new_state_h, new_state_c]
        # self.decoder_model = Model(
        #     inputs=[decoder_inputs_inf,
        #             encoder_outputs_input,
        #             decoder_state_input_h,
        #             decoder_state_input_c],
        #     outputs=[decoder_outputs_final_inf, state_h_inf2, state_c_inf2],
        #     name='decoder_model_inference'
        # )

        # print("Inference models built successfully!")


#     def train_model(self, epochs=None, batch_size=None, validation_split=0.1):
#         """Train the model with improved training loop"""
#         if epochs is None:
#             epochs = self.epochs
#         if batch_size is None:
#             batch_size = self.batch_size
            
#         print(f"Training model for {epochs} epochs with batch size {batch_size}")
        
#         if not self.model:
#             self.build_model()
        
#         # Improved callbacks
#         callbacks = [
#             EarlyStopping(
#                 monitor='val_loss',
#                 patience=8,
#                 restore_best_weights=True,
#                 verbose=1,
#                 min_delta=0.001
#             ),
#             ReduceLROnPlateau(
#                 monitor='val_loss',
#                 factor=0.5,
#                 patience=4,
#                 min_lr=1e-6,
#                 verbose=1,
#                 cooldown=2
#             ),
#             ModelCheckpoint(
#                 filepath=f"{self.checkpoint_dir}/best_model.h5",
#                 monitor='val_loss',
#                 save_best_only=True,
#                 save_weights_only=False,
#                 verbose=1
#             ),
#             CSVLogger(
#                 filename=f"{self.logs_dir}/training.csv",
#                 append=True
#             )
#         ]
        
#         # Train the model
#         history = self.model.fit(
#             [self.preprocessor.splits['train']['questions'], 
#              self.preprocessor.splits['train']['answer_decoder_input']],
#             self.preprocessor.splits['train']['answer_decoder_target'],
#             validation_data=(
#                 [self.preprocessor.splits['val']['questions'], 
#                  self.preprocessor.splits['val']['answer_decoder_input']],
#                 self.preprocessor.splits['val']['answer_decoder_target']
#             ),
#             epochs=epochs,
#             batch_size=batch_size,
#             callbacks=callbacks,
#             verbose=1,
#             shuffle=True
#         )
        
#         # Save final model
#         final_model_path = f"{self.model_dir}/final_model.h5"
#         self.model.save(final_model_path)
#         print(f"Final model saved to: {final_model_path}")
        
#         # Rebuild inference models after training
#         self._build_inference_models()
        
#         return history

#     def predict_answer_improved(self, question_text, max_length=None, temperature=0.7, top_k=40, top_p=0.9):
        # """Improved answer prediction with better decoding strategies."""
        # if not self.model or not self.encoder_model or not self.decoder_model:
        #     raise ValueError("Model not trained or inference models not built!")

        # if max_length is None:
        #     max_length = min(self.preprocessor.max_answer_length, 100)

        # # 1) Tokenize & pad the question
        # question_seq = self.preprocessor.tokenizer_questions.texts_to_sequences([question_text.lower()])
        # if not question_seq or not question_seq[0]:
        #     print("Warning: Question could not be tokenized properly")
        #     return "I couldn't understand the question."

        # question_padded = pad_sequences(
        #     question_seq,
        #     maxlen=self.preprocessor.max_question_length,
        #     padding='post',
        #     truncating='post'
        # )

        # # 2) Run the encoder → three outputs: (enc_outputs, h, c)
        # encoder_outputs, state_h, state_c = self.encoder_model.predict(question_padded, verbose=0)
        # encoder_states = [state_h, state_c]  # both have shape (1, feat_dim)

        # # 3) Start decoder with <START> token (assumed index 1)
        # target_seq = np.zeros((1, 1), dtype='int32')
        # target_seq[0, 0] = 1  # <START> index

        # decoded_tokens = []
        # states_value = encoder_states

        # for _ in range(max_length):
        #     output_tokens, h, c = self.decoder_model.predict(
        #         [target_seq, encoder_outputs, states_value[0], states_value[1]],
        #         verbose=0
        #     )

        #     # output_tokens.shape == (1, 1, vocab_size)
        #     preds = output_tokens[0, 0, :] / max(temperature, 1e-9)

        #     # … (top‐k, top‐p, softmax, sampling logic unchanged) …

        #     if sampled_token_index in [0, 2]:  # if <PAD> or <END>
        #         break

        #     decoded_tokens.append(sampled_token_index)
        #     target_seq = np.zeros((1, 1), dtype='int32')
        #     target_seq[0, 0] = sampled_token_index
        #     states_value = [h, c]

        # # 5) Convert tokens back to text (unchanged)
        # if decoded_tokens:
        #     try:
        #         reverse_word_index = {v: k for k, v in self.preprocessor.tokenizer_answers.word_index.items()}
        #         reverse_word_index[0] = ''
        #         reverse_word_index[1] = '<START>'
        #         reverse_word_index[2] = '<END>'

        #         decoded_words = []
        #         for tok in decoded_tokens:
        #             if tok > 2 and tok in reverse_word_index:
        #                 w = reverse_word_index[tok]
        #                 if w not in ['<START>', '<END>', '<UNK>']:
        #                     decoded_words.append(w)

        #         if decoded_words:
        #             answer = ' '.join(decoded_words)
        #             return self._clean_generated_text(answer)
        #         else:
        #             return "I couldn't generate a proper answer."
        #     except Exception as e:
        #         print(f"Error in text conversion: {e}")
        #         return "Error in generating answer."
        # else:
        #     return "No answer generated."

#     def _clean_generated_text(self, text):
#         """Clean and improve generated text"""
#         if not text:
#             return "No answer available."
            
#         # Remove extra spaces and clean up
#         text = ' '.join(text.split())
        
#         # Capitalize first letter
#         if text:
#             text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
        
#         # Add period if not present
#         if text and not text[-1] in '.!?':
#             text += '.'
            
#         return text

#     def generate_answer(self, question_text):
#         """Main method to generate answer with fallback to LLM"""
#         try:
#             # First check if it's cybersecurity related
#             is_cybersec = False
#             if self.llm_client:
#                 is_cybersec = self.llm_client.is_cybersecurity_related(question_text)
            
#             # Try LSTM prediction first
#             lstm_answer = self.predict_answer_improved(
#                 question_text, 
#                 max_length=80,
#                 temperature=0.8,
#                 top_k=50,
#                 top_p=0.9
#             )
            
#             # Check if LSTM answer is good enough
#             if (lstm_answer and 
#                 len(lstm_answer.split()) > 3 and 
#                 not self._is_gibberish(lstm_answer)):
#                 return f"LSTM: {lstm_answer}"
            
#             # Fallback to LLM if available and cybersecurity related
#             if self.llm_client and is_cybersec:
#                 llm_answer = self.llm_client.generate_answer(question_text)
#                 return f"LLM: {llm_answer}"
            
#             # Final fallback
#             return "I'm sorry, I couldn't generate a proper answer for this question."
            
#         except Exception as e:
#             print(f"Error in generate_answer: {e}")
#             return "An error occurred while generating the answer."

#     def _is_gibberish(self, text):
#         """Check if generated text is gibberish"""
#         if not text or len(text.strip()) < 3:
#             return True
            
#         words = text.split()
#         if len(words) < 2:
#             return True
            
#         # Check for repeated words
#         unique_words = set(words)
#         if len(unique_words) / len(words) < 0.5:  # Too many repeated words
#             return True
            
#         # Check for very short words (possible tokenization issues)
#         avg_word_length = sum(len(word) for word in words) / len(words)
#         if avg_word_length < 2.5:
#             return True
            
#         return False

#     def save_model(self, filepath=None):
#         """Save the complete model in native Keras format (.keras)."""
#         if filepath is None:
#             filepath = f"{self.model_dir}/complete_model.keras"
#         if self.model:
#             # Calling model.save with a `.keras` extension will use the new native-keras format automatically.
#             self.model.save(filepath)
#             print(f"Model saved to: {filepath}")
#         else:
#             print("No model to save!")

#     def load_model(self, filepath=None):
#         """Load a trained model from the native Keras `.keras` format."""
#         if filepath is None:
#             filepath = f"{self.model_dir}/complete_model.keras"
#         try:
#             # This will correctly deserialize any built-in TensorFlow ops (including NotEqual)
#             self.model = load_model(filepath)
#             # Re-build inference sub-models after loading
#             self._build_inference_models()
#             print(f"Model loaded from: {filepath}")
#             return True
#         except Exception as e:
#             print(f"Error loading model: {e}")
#             return False
    
#     def evaluate_model(self):
#         """Evaluate model performance"""
#         if not self.model:
#             print("No model to evaluate!")
#             return None
            
#         print("Evaluating model on test set...")
        
#         test_loss, test_acc = self.model.evaluate(
#             [self.preprocessor.splits['test']['questions'], 
#              self.preprocessor.splits['test']['answer_decoder_input']],
#             self.preprocessor.splits['test']['answer_decoder_target'],
#             batch_size=self.batch_size,
#             verbose=1
#         )
        
#         print(f"Test Loss: {test_loss:.4f}")
#         print(f"Test Accuracy: {test_acc:.4f}")
        
#         # Test with sample questions
#         print("\n--- Sample Predictions ---")
#         sample_questions = [
#             "What is cybersecurity?",
#             "How does encryption work?",
#             "What is a firewall?",
#             "Explain malware detection",
#             "What is network security?"
#         ]
        
#         for question in sample_questions:
#             answer = self.predict_answer_improved(question, max_length=50)
#             print(f"Q: {question}")
#             print(f"A: {answer}")
#             print("-" * 50)
        
#         return test_loss, test_acc

# # Enhanced training function with better parameters
# def train_improved_model():
#     """Train the improved LSTM model with better settings"""
    
#     # Load preprocessor
#     preprocessor_path = "./preprocessed_data"
#     print("Loading preprocessor...")
#     preprocessor = ImprovedLSTMQAPreprocessorLoader(preprocessor_path)
    
#     # Initialize LLM client (optional)
#     llm_client = None
#     try:
#         groq_api_key = os.getenv('GROQ_API_KEY') or "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"
#         if groq_api_key:
#             llm_client = CybersecurityLLM(api_key=groq_api_key)
#             print("LLM client initialized successfully!")
#         else:
#             print("No GROQ_API_KEY found, LLM fallback disabled")
#     except Exception as e:
#         print(f"LLM initialization failed: {e}")
    
#     # Initialize model with better hyperparameters
#     model = ImprovedLSTMModel(
#         preprocessor=preprocessor, 
#         llm_client=llm_client,
#         model_name="cybersec_lstm_v2"
#     )
    
#     # Adjust hyperparameters for better text generation
#     model.embedding_dim = 300   # Increased embedding dimension
#     model.encoder_units = 256   # Balanced encoder units
#     model.decoder_units = 256   # Balanced decoder units
#     model.attention_units = 128 # Appropriate attention units
#     model.dropout_rate = 0.3    # Increased dropout to prevent overfitting
#     model.learning_rate = 0.001 # Slightly higher learning rate
#     model.batch_size = 16       # Smaller batch size for better convergence
#     model.epochs = 30           # Reasonable number of epochs
    
#     print("Building improved model...")
#     model.build_model()   # <-- this also builds encoder_model & decoder_model
    
#     # Print model summary
#     print("\n--- Model Architecture ---")
#     model.model.summary()
    
#     print("\n--- Starting Training ---")
#     history = model.train_model(epochs=model.epochs, batch_size=model.batch_size)
    
#     print("\n--- Training Completed ---")
    
#     # Evaluate the model
#     model.evaluate_model()
    
#     # Save the final model
#     model.save_model()
    
#     return model, history

# # Function to test the trained model
# def test_model_predictions():
#     """Test the trained model with various questions"""
    
#     # Load preprocessor
#     preprocessor_path = "./preprocessed_data"
#     preprocessor = ImprovedLSTMQAPreprocessorLoader(preprocessor_path)
    
#     # Initialize model
#     model = ImprovedLSTMModel(preprocessor=preprocessor, model_name="cybersec_lstm_v2")
    
#     # Load trained model
#     if model.load_model():
#         print("Testing model predictions...")
        
#         test_questions = [
#             "What is a DDoS attack?",
#             "How do you prevent SQL injection?",
#             "What is multi-factor authentication?",
#             "Explain network segmentation",
#             "What are the types of malware?",
#             "How does SSL encryption work?",
#             "What is a vulnerability assessment?",
#             "Describe incident response process",
#             "What is zero-day exploit?",
#             "How to secure cloud infrastructure?"
#         ]
        
#         print("\n" + "="*80)
#         print("MODEL PREDICTION TESTS")
#         print("="*80)
        
#         for i, question in enumerate(test_questions, 1):
#             print(f"\n[{i:2d}] Question: {question}")
#             print("-" * 60)
            
#             try:
#                 answer = model.generate_answer(question)
#                 print(f"Answer: {answer}")
#             except Exception as e:
#                 print(f"Error: {e}")
            
#             print("-" * 60)
#     else:
#         print("Could not load trained model. Please train the model first.")

# if __name__ == "__main__":
#     import sys
    
#     if len(sys.argv) > 1:
#         if sys.argv[1] == "train":
#             print("Starting model training...")
#             model, history = train_improved_model()
#             print("Training completed!")
            
#         elif sys.argv[1] == "test":
#             print("Testing trained model...")
#             test_model_predictions()
            
#         elif sys.argv[1] == "both":
#             print("Training then testing model...")
#             model, history = train_improved_model()
#             print("\nNow testing the trained model...")
#             test_model_predictions()
            
#         else:
#             print("Usage: python train.py [train|test|both]")
#     else:
#         print("Usage: python train.py [train|test|both]")
#         print("Options:")
#         print("  train - Train the model")
#         print("  test  - Test existing trained model")
#         print("  both  - Train then test the model")





