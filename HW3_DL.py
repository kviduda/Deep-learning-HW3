import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast, BertForQuestionAnswering
from torch.optim import AdamW
import json
from tqdm import tqdm
from sklearn.metrics import f1_score
import os

# Handle tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data with a limit on the number of examples
def get_data(path, limit=None):
    with open(path, 'r') as f:
        raw_data = json.load(f)
    contexts, questions, answers = [], [], []
    for group in raw_data['data']:
        for paragraph in group['paragraphs']:
            context = paragraph['context'].lower()
            for qa in paragraph['qas']:
                questions.append(qa['question'].lower())
                answers.append({'text': qa['answers'][0]['text'].lower(),
                                'answer_start': qa['answers'][0]['answer_start']})
                contexts.append(context)
                if limit is not None and len(contexts) >= limit:
                    return contexts, questions, answers
    return contexts, questions, answers

train_data_path = 'spoken_train-v1.1.json'
test_data_path = 'spoken_test-v1.1.json'
train_contexts, train_questions, train_answers = get_data(train_data_path, limit=500)
valid_contexts, valid_questions, valid_answers = get_data(test_data_path, limit=100)

# Add end index for answers
def add_end_idx(answers, contexts):
    for answer, context in zip(answers, contexts):
        gold_text = answer['text']
        start_idx = answer['answer_start']
        end_idx = start_idx + len(gold_text)
        answer['answer_end'] = end_idx

add_end_idx(train_answers, train_contexts)
add_end_idx(valid_answers, valid_contexts)

# Tokenization
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
train_encodings = tokenizer(train_contexts, train_questions, truncation=True, padding=True, max_length=512)
valid_encodings = tokenizer(valid_contexts, valid_questions, truncation=True, padding=True, max_length=512)

# Adjust token positions for model inputs
def add_token_positions(encodings, answers):
    start_positions, end_positions = [], []
    for i, answer in enumerate(answers):
        start_pos = encodings.char_to_token(i, answer['answer_start'])
        end_pos = encodings.char_to_token(i, answer['answer_end'])

        if start_pos is None:
            start_pos = tokenizer.model_max_length
        if end_pos is None:
            end_pos = tokenizer.model_max_length

        start_positions.append(start_pos)
        end_positions.append(end_pos)

    encodings.update({'start_positions': start_positions, 'end_positions': end_positions})

add_token_positions(train_encodings, train_answers)
add_token_positions(valid_encodings, valid_answers)

# Dataset class
class SQuADDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

# Data loaders
train_dataset = SQuADDataset(train_encodings)
valid_dataset = SQuADDataset(valid_encodings)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)

# Model and optimizer
model = BertForQuestionAnswering.from_pretrained("bert-base-uncased").to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)

# Mixed precision setup
scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

# Training function
def train(model, dataloader, scaler):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)

        optimizer.zero_grad()
        if scaler:
            with torch.cuda.amp.autocast():
                outputs = model(input_ids, attention_mask=attention_mask,
                                start_positions=start_positions, end_positions=end_positions)
                loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(input_ids, attention_mask=attention_mask,
                            start_positions=start_positions, end_positions=end_positions)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
        total_loss += loss.item()

    print(f"Train Loss: {total_loss / len(dataloader)}")

# Evaluation function
def evaluate(model, dataloader):
    model.eval()
    start_true, start_pred, end_true, end_pred = [], [], [], []
    for batch in tqdm(dataloader, desc="Evaluating"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits

        start_preds = torch.argmax(start_logits, dim=1)
        end_preds = torch.argmax(end_logits, dim=1)

        start_true.extend(start_positions.cpu().numpy())
        end_true.extend(end_positions.cpu().numpy())
        start_pred.extend(start_preds.cpu().numpy())
        end_pred.extend(end_preds.cpu().numpy())

    start_f1 = f1_score(start_true, start_pred, average='macro')
    end_f1 = f1_score(end_true, end_pred, average='macro')

    print(f"Start F1 Score: {start_f1}")
    print(f"End F1 Score: {end_f1}")
    print(f"Average F1 Score: {((start_f1 + end_f1) / 2)}")

# Running training and evaluation over multiple epochs
EPOCHS = 3
for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}")
    train(model, train_loader, scaler)
    evaluate(model, valid_loader)
