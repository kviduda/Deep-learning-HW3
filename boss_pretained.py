import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast, BertForQuestionAnswering
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
import json
from tqdm import tqdm
import os

# Prevent potential deadlocks from tokenizer parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set device based on CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and prepare data
def get_data(path, limit=None):
    with open(path, 'rb') as f:
        raw_data = json.load(f)
    con, que, answ, starts, ends = [], [], [], [], []

    count = 0
    for group in raw_data['data']:
        for paragraph in group['paragraphs']:
            context = paragraph['context'].lower()
            for qa in paragraph['qas']:
                question = qa['question'].lower()
                for answer in qa['answers']:
                    answer_text = answer['text'].lower()
                    start_idx = context.find(answer_text)
                    if start_idx != -1:  # Ensure the answer is found within the context
                        con.append(context)
                        que.append(question)
                        answ.append(answer_text)
                        starts.append(start_idx)
                        ends.append(start_idx + len(answer_text))
                        count += 1
                        if limit and count >= limit:
                            break
                if limit and count >= limit:
                    break
        if limit and count >= limit:
            break
    return con, que, answ, starts, ends

train_data_path = 'spoken_train-v1.1.json'
test_data_path = 'spoken_test-v1.1.json'
train_contexts, train_questions, train_answers, train_starts, train_ends = get_data(train_data_path, limit=1000)
valid_contexts, valid_questions, valid_answers, valid_starts, valid_ends = get_data(test_data_path, limit=200)

# Tokenizer and Model Initialization
MODEL_PATH = "deepset/bert-base-cased-squad2"
tokenizerFast = BertTokenizerFast.from_pretrained(MODEL_PATH)
model = BertForQuestionAnswering.from_pretrained(MODEL_PATH).to(device)

# Encode data with doc_stride
def encode_data(tokenizer, questions, contexts, answers, starts, ends, max_length=256, doc_stride=128):
    encodings = tokenizer(
        questions, contexts, max_length=max_length, truncation=True, padding='max_length', return_offsets_mapping=True, 
        stride=doc_stride
    )
    start_positions = [adjust_positions(pos, encodings.offset_mapping[i]) for i, pos in enumerate(starts)]
    end_positions = [adjust_positions(pos, encodings.offset_mapping[i], is_end=True) for i, pos in enumerate(ends)]

    encodings.update({'start_positions': start_positions, 'end_positions': end_positions})
    del encodings['offset_mapping']  # We don't want to return offset mapping to the model
    return encodings

def adjust_positions(pos, offsets, is_end=False):
    # Converts character position to token position
    for i, (start, end) in enumerate(offsets):
        if start <= pos < end or (is_end and start < pos <= end):
            return i
    return -1  # Return an invalid index for positions that cannot be mapped (error handling)

train_encodings = encode_data(tokenizerFast, train_questions, train_contexts, train_answers, train_starts, train_ends, doc_stride=128)
valid_encodings = encode_data(tokenizerFast, valid_questions, valid_contexts, valid_answers, valid_starts, valid_ends, doc_stride=128)

# Dataset Class
class QA_Dataset(Dataset):
    def __init__(self, encodings, answers=None):
        self.encodings = encodings
        self.answers = answers

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.answers is not None:
            item['answers'] = self.answers[idx]  # Include answers for validation
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

train_dataset = QA_Dataset(train_encodings)
valid_dataset = QA_Dataset(valid_encodings, valid_answers)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=1)
valid_loader = DataLoader(valid_dataset, batch_size=4, num_workers=1)

# Optimizer and Gradient Scaler Initialization
optimizer = AdamW(model.parameters(), lr=5e-5)
scaler = GradScaler() if device.type == 'cuda' else None

# Custom F1 Score Calculation
def calculate_f1(prediction, reference):
    pred_tokens = prediction.split()
    ref_tokens = reference.split()
    common_tokens = set(pred_tokens) & set(ref_tokens)
    
    if len(common_tokens) == 0:
        return 0.0
    
    precision = len(common_tokens) / len(pred_tokens)
    recall = len(common_tokens) / len(ref_tokens)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

# Training and Validation Functions
def train_model(model, dataloader, optimizer, scaler, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device, non_blocking=True)
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)

        if scaler:
            with autocast():
                outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                start_positions=start_positions, end_positions=end_positions)
                loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                            start_positions=start_positions, end_positions=end_positions)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def validate_model(model, dataloader, device):
    model.eval()
    total_f1 = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            start_pred = torch.argmax(outputs.start_logits, dim=1)
            end_pred = torch.argmax(outputs.end_logits, dim=1)

            predicted_answers = [
                tokenizerFast.convert_tokens_to_string(
                    tokenizerFast.convert_ids_to_tokens(input_ids[i][start_pred[i]:end_pred[i] + 1]))
                for i in range(len(start_pred))
            ]
            references = batch['answers']  # Ensure 'answers' key exists
            
            for pred, ref in zip(predicted_answers, references):
                f1 = calculate_f1(pred, ref)
                total_f1 += f1

    return total_f1 / len(dataloader)

# Training Loop
if __name__ == "__main__":
    EPOCHS = 3
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}")
        train_loss = train_model(model, train_loader, optimizer, scaler, device)
        print(f"Training Loss: {train_loss:.4f}")
        f1_score = validate_model(model, valid_loader, device)
        print(f"Validation F1 Score: {f1_score:.4f}")
