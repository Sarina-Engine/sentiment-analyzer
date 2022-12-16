import hazm
from cleantext import clean
import re

class Preprocessor:
    def __init__(self):
        self.minlim = 3
        self.maxlim = 256

    def __call__(self, text):
        # cleaning comments
        cleaned_text = self.cleaning(text)
        return cleaned_text

    def normalize_comments_length(self, text):
        # calculate the length of comments based on their words
        comment_word_count =  len(hazm.word_tokenize(text))

        # remove comments with the length of fewer than three words
        if self.minlim < comment_word_count <= self.maxlim:
            text = ''

        return text

    def cleanhtml(self, raw_html):
        cleanr = re.compile('<.*?>')
        cleantext = re.sub(cleanr, '', raw_html)
        return cleantext


    def cleaning(self, text):
        text = text.strip()
        
        # regular cleaning
        text = clean(text,
            fix_unicode=True,
            to_ascii=False,
            lower=True,
            no_line_breaks=True,
            no_urls=True,
            no_emails=True,
            no_phone_numbers=True,
            no_numbers=False,
            no_digits=False,
            no_currency_symbols=True,
            no_punct=False,
            replace_with_url="",
            replace_with_email="",
            replace_with_phone_number="",
            replace_with_number="",
            replace_with_digit="0",
            replace_with_currency_symbol="",
        )

        # cleaning htmls
        text = self.cleanhtml(text)
        
        # normalizing
        normalizer = hazm.Normalizer()
        text = normalizer.normalize(text)
        
        # removing wierd patterns
        wierd_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            u"\U0001f926-\U0001f937"
            u'\U00010000-\U0010ffff'
            u"\u200d"
            u"\u2640-\u2642"
            u"\u2600-\u2B55"
            u"\u23cf"
            u"\u23e9"
            u"\u231a"
            u"\u3030"
            u"\ufe0f"
            u"\u2069"
            u"\u2066"
            # u"\u200c"
            u"\u2068"
            u"\u2067"
            "]+", flags=re.UNICODE)
        
        text = wierd_pattern.sub(r'', text)
        
        # removing extra spaces, hashtags
        text = re.sub("#", "", text)
        text = re.sub("\s+", " ", text)
        
        return text
