import sacrebleu
from jiwer import wer
from pathlib import Path

# Resolve paths relative to this script's folder (more reliable than PyCharm's working directory).
BASE_DIR = Path(__file__).resolve().parent
HUMAN_PATH = BASE_DIR / "human_translation.txt"
LLM_PATH = BASE_DIR / "llm_translation.txt"

# Read full documents from the .txt files
human_text = HUMAN_PATH.read_text(encoding="utf-8").strip()
llm_text = LLM_PATH.read_text(encoding="utf-8").strip()

# SacreBLEU expects lists (corpus), even for single documents
human_translation = [human_text]
llm_translation = [llm_text]

# Calculates BLEU (from SacreBLEU) =====
bleu = sacrebleu.corpus_bleu(llm_translation, [human_translation])
print(f"BLEU: {bleu.score:.2f}")

# Calculates WER (Word Error Rate, from JiWER) =====
wer_score = wer(human_text, llm_text)
print(f"WER: {wer_score * 100:.2f}%")
