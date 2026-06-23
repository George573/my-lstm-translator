import pickle
import re
import json
from tqdm import tqdm
from typing import List, Tuple, Set

def load_rules_tokens(rules_file: str, tokens_file: str) -> Tuple[List, List]:
    with open(rules_file, 'rb') as f_rules, open(tokens_file, 'rb') as f_tokens:
        return pickle.load(f_rules), pickle.load(f_tokens)

class Tokenizer:
    def __init__(self, rules: List[List[str]], tokens: List[str], ukn_token: str = '<ukn>'):
        self.ukn_token_ = ukn_token
        
        self.valid_chars_: Set[str] = set(tokens)
        self.words_cache_: dict = {}
        
        self.word_split_regex_ = re.compile(r'[^\W_]+(?:[\'\-][^\W_]+)*\'?')
        
        self.rule_strings_: List[str] = [' '.join(rule) for rule in rules]
        self.rule_replacements_: List[str] = [''.join(rule) for rule in rules]
        
        self.rules_compiled_: List[re.Pattern] = [
            re.compile(r'(?<!\S)' + re.escape(rule_str) + r'(?!\S)') 
            for rule_str in self.rule_strings_
        ]
    
    # For the newer version of bpe impl
    @classmethod
    def load(cls, filepath: str, ukn_token: str = '<ukn>') -> "Tokenizer":
        """Loads the tokenizer using the new JSON format."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Extract rules and alphabet from the JSON saved by the trainer
        rules = data.get('rules', [])
        tokens = data.get('tokens', [])
        ukn_token_ = data.get('ukn_token', [])
        
        ukn_token = ukn_token_ if ukn_token_ else ukn_token
        
        assert rules, 'No rules found in json file'
        assert tokens, 'No tokens found in json file'
        
        return cls(rules=rules, tokens=tokens, ukn_token=ukn_token)

    def split_to_words_sparse(self, texts: List[str]) -> List[List[str]]:
        valid_chars = self.valid_chars_
        word_regex = self.word_split_regex_
        all_words = []
        
        for text in texts:
            words = word_regex.findall(text)
            
            processed_words = [
                ' '.join(char for char in word if char in valid_chars) + ' </w>'
                for word in words
            ] 
            all_words.append(processed_words)
            
        return all_words

    def apply_rules(self, word: str) -> List[str]:
        if word in self.words_cache_:
            return self.words_cache_[word]
        
        original_word = word
        
        for rule_comp, replacement in zip(
            self.rules_compiled_, self.rule_replacements_
        ):
            word = rule_comp.sub(replacement, word)
        
        tokenized_word = word.split()
        
        self.words_cache_[original_word] = tokenized_word
        
        return tokenized_word

    def tokenize_text(self, words: List[str]) -> List[str]:
        tokenized_words = []
        for word in words:
            tokenized_words.extend(self.apply_rules(word))
        return tokenized_words

    def tokenize_texts(self, texts: List[str]) -> List[List[str]]:
        return [
            self.tokenize_text(words) 
            for words in self.split_to_words_sparse(texts)
        ]