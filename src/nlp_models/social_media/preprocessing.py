"""Optional preprocessing and augmentation used in the social-media comparison."""

from __future__ import annotations


def build_social_media_preprocessor():
    """Build the Ekphrasis pipeline used for the preprocessing run."""
    try:
        from ekphrasis.classes.preprocessor import TextPreProcessor
        from ekphrasis.classes.tokenizer import SocialTokenizer
        from ekphrasis.dicts.emoticons import emoticons
    except ImportError as exc:
        raise ImportError("Install ekphrasis to use social-media preprocessing.") from exc

    return TextPreProcessor(
        normalize=["url", "email", "percent", "money", "phone", "user", "time", "date", "number"],
        annotate={},
        fix_html=True,
        segmenter="twitter",
        corrector="twitter",
        unpack_hashtags=True,
        unpack_contractions=True,
        spell_correct_elong=False,
        tokenizer=SocialTokenizer(lowercase=True).tokenize,
        dicts=[emoticons],
    )


def preprocess_social_texts(texts, processor=None):
    processor = processor or build_social_media_preprocessor()
    return [" ".join(processor.pre_process_doc(text)) for text in texts]


def augment_social_texts(texts):
    """Return synonym-replaced and random-deletion variants using nlpaug."""
    try:
        import nlpaug.augmenter.word as naw
    except ImportError as exc:
        raise ImportError("Install nlpaug and its NLTK resources to use augmentation.") from exc

    synonym = naw.SynonymAug(aug_src="wordnet")
    deletion = naw.RandomWordAug(action="delete")

    def as_text(value):
        return value[0] if isinstance(value, list) else value

    synonym_texts = [as_text(synonym.augment(text)) for text in texts]
    deleted_texts = [as_text(deletion.augment(text)) for text in texts]
    return synonym_texts, deleted_texts
