from typing import List, Tuple, Type
import re
from collections import Counter
import math
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from pydantic import BaseModel, PrivateAttr
from crewai.tools import BaseTool

# Download necessary NLTK data if not present
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# --------- Input Schema ---------
class DocumentAnalyzeArgs(BaseModel):
    text: str

# --------- Tool Implementation ---------
class DocumentAnalyzerTool(BaseTool):
    name: str = "analyze_document"
    description: str = "Comprehensively analyze document content to extract key information."
    args_schema: Type[DocumentAnalyzeArgs] = DocumentAnalyzeArgs
    _stop_words: set = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._stop_words = set(stopwords.words('english'))

    def _run(self, text: str) -> str:
        try:
            if not text or len(text.strip()) == 0:
                return "Error: Empty document provided."
            word_count = len(re.findall(r'\w+', text))
            sentence_count = len(sent_tokenize(text))
            paragraph_count = len(text.split('\n\n'))
            avg_sentence_length = word_count / max(1, sentence_count)

            key_phrases = self._extract_key_phrases(text)
            sentiment_score, sentiment = self._analyze_sentiment(text)
            entities = self._extract_entities(text)

            analysis = f"""
Document Analysis Report
========================
Basic Statistics:
-----------------
Word count: {word_count}
Sentence count: {sentence_count}
Paragraph count: {paragraph_count}
Average sentence length: {avg_sentence_length:.1f} words

Content Summary:
----------------
Key phrases:
"""
            for i, (phrase, score) in enumerate(key_phrases[:5], 1):
                analysis += f"{i}. {phrase} (relevance: {score:.2f})\n"

            analysis += f"\nSentiment: {sentiment} (score: {sentiment_score:.2f})\n"

            if entities:
                analysis += "\nPotential entities:\n"
                for i, entity in enumerate(entities[:10], 1):
                    analysis += f"{i}. {entity}\n"

            key_sentences = self._extract_key_sentences(text, key_phrases)
            if key_sentences:
                analysis += "\nKey content:\n"
                for i, sentence in enumerate(key_sentences[:3], 1):
                    analysis += f"{i}. {sentence}\n"

            return analysis

        except Exception as e:
            return f"Error analyzing document: {str(e)}"

    def _extract_key_phrases(self, text: str, max_phrases: int = 10) -> List[Tuple[str, float]]:
        words = word_tokenize(text.lower())
        words = [word for word in words if word.isalnum() and word not in self._stop_words]
        word_freq = Counter(words)
        word_scores = {
            word: count * math.log(len(words) / (count + 1) + 1)
            for word, count in word_freq.items()
        }
        sentences = sent_tokenize(text)
        phrases = []
        for sentence in sentences:
            sentence_words = word_tokenize(sentence.lower())
            for i in range(len(sentence_words)):
                word = sentence_words[i]
                if word.isalnum() and word not in self._stop_words:
                    phrases.append((word, word_scores.get(word, 0)))
            for i in range(len(sentence_words) - 1):
                if (sentence_words[i].isalnum() and
                    sentence_words[i+1].isalnum() and
                    sentence_words[i] not in self._stop_words):
                    bigram = f"{sentence_words[i]} {sentence_words[i+1]}"
                    score = ((word_scores.get(sentence_words[i], 0) +
                              word_scores.get(sentence_words[i+1], 0)) / 2) * 1.5
                    phrases.append((bigram, score))
        sorted_phrases = sorted(phrases, key=lambda x: x[1], reverse=True)
        unique_phrases = []
        for phrase, score in sorted_phrases:
            if len(unique_phrases) >= max_phrases:
                break
            is_subphrase = any(phrase in existing for existing, _ in unique_phrases)
            if not is_subphrase:
                unique_phrases.append((phrase, score))
        return unique_phrases

    def _analyze_sentiment(self, text: str) -> Tuple[float, str]:
        positive_words = [
            'good', 'great', 'excellent', 'positive', 'best', 'innovative',
            'impressive', 'helpful', 'beneficial', 'advantage', 'success',
            'happy', 'pleased', 'effective', 'useful', 'better', 'remarkable'
        ]
        negative_words = [
            'bad', 'poor', 'negative', 'worst', 'problem', 'issue',
            'disappointing', 'difficult', 'failure', 'concern', 'weakness',
            'disadvantage', 'trouble', 'ineffective', 'useless', 'worse'
        ]
        words = word_tokenize(text.lower())
        pos_count = sum(1 for word in words if word in positive_words)
        neg_count = sum(1 for word in words if word in negative_words)
        total = pos_count + neg_count
        score = (pos_count - neg_count) / total if total > 0 else 0
        if score > 0.2:
            sentiment = "positive"
        elif score < -0.2:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        return score, sentiment

    def _extract_entities(self, text: str) -> List[str]:
        sentences = sent_tokenize(text)
        entities = set()
        for sentence in sentences:
            words = word_tokenize(sentence)
            for i in range(1, len(words)):
                word = words[i]
                if word[0].isupper() and word.isalpha() and word.lower() not in self._stop_words:
                    if i < len(words) - 1 and words[i+1][0].isupper():
                        entities.add(f"{word} {words[i+1]}")
                    else:
                        entities.add(word)
        return list(entities)

    def _extract_key_sentences(self, text: str, key_phrases: List[Tuple[str, float]]) -> List[str]:
        sentences = sent_tokenize(text)
        sentence_scores = []
        phrase_dict = {phrase.lower(): score for phrase, score in key_phrases}
        for sentence in sentences:
            score = 0
            for phrase, phrase_score in key_phrases:
                if phrase.lower() in sentence.lower():
                    score += phrase_score
            word_count = len(word_tokenize(sentence))
            if word_count < 5:
                score *= 0.5
            elif word_count > 25:
                score *= 0.8
            sentence_scores.append((sentence, score))
        sorted_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)
        return [sentence for sentence, score in sorted_sentences[:5]]
