from datasets import Dataset
import fitz
import faiss
import json
import os
import pandas as pd
from pathlib import Path
import pickle
import numpy as np
import re
import sentence_transformers
import spacy
from tqdm.auto import tqdm
from typing import Union, List
from huggingface_hub import upload_file
from huggingface_hub import HfApi, upload_file
from huggingface_hub.utils import RepositoryNotFoundError
import transformers

def get_files(root: Union[Path, str], extension: str = '') -> List[Path]:
    """Get all files with given extension"""
    files = sorted([file for file in Path(root).rglob(f'*{extension}') if file.is_file()])
    return files

def load_json(path: str):
    return json.loads(Path(path).read_text())

class Loader:

    """
    Load documents from disk and normalize them into a paragraph-level DataFrame.

    The Loader supports PDF, plain text, and Parquet inputs. Text is extracted
    with minimal normalization to preserve word boundaries and surface form,
    which is important for downstream tasks such as span alignment, highlighting,
    or embedding-based retrieval.

    Each loaded document is represented as a pandas DataFrame with at least:
        - document_id : str
            A stable identifier for the source document.
        - paragraph : str
            A text segment extracted from the document.

    Parameters
    ----------
    input_path : str
        Path to the input document (.pdf, .txt, or .parquet).
    **kwargs
        Optional keyword arguments.
        - document_id : str, optional
            Explicit document identifier. Defaults to the input file stem.

    Notes
    -----
    - PDF extraction uses block-level text from PyMuPDF and performs only
      minimal cleaning (e.g., removing soft hyphens) to avoid altering
      original token sequences.
    - Line joins in PDFs may introduce spaces within words split across lines.
      This behavior is intentional and may be refined at later pipeline stages.
    - The `load()` method populates `self.paragraphs` as a side effect.
    """

    def __init__(
        self,
        input_path: str,
        **kwargs
    ):
        self.input = Path(input_path)
        self.document_id = kwargs.get("document_id", self.input.stem)

    def clean_pdf_text(self, text: str) -> str:
        """
        Clean PDF text minimally so words are preserved exactly
        as they appear in the PDF for reliable highlight matching.
        """
        # Normalize carriage returns
        text = text.replace("\r", "\n")
        # Remove soft hyphens (U+00AD) that break words visually
        text = text.replace("\u00ad", "")
        # Keep all newlines; only collapse excessive spaces (not touching newlines)
        text = re.sub(r'[ ]{2,}', ' ', text)
        return text.strip()

    def load_pdf(self) -> pd.DataFrame:
        """
        Extract text in blocks while preserving word boundaries and layout.
        """
        print(f"Building dataframe from {self.input}")

        doc = fitz.open(self.input)
        page_nums = list(range(len(doc)))
        records = []

        for page_idx in tqdm(page_nums, desc="Extracting text blocks"):
            page = doc.load_page(page_idx)
            page_dict = page.get_text("dict")

            for block in page_dict["blocks"]:
                if "lines" not in block:
                    continue  # skip images or non-text blocks

                block_text_lines = []
                for line in block["lines"]:
                    # Join all spans in the line
                    line_text = "".join(span["text"] for span in line["spans"])
                    line_text = self.clean_pdf_text(line_text)
                    block_text_lines.append(line_text)

                # WARNING: This will end up creating spaces within words that are split across lines. 
                block_text = " ".join(block_text_lines)
                if block_text.strip():
                    records.append({
                        "document_id": self.document_id,
                        "paragraph": block_text
                    })

        df = pd.DataFrame.from_records(records)
        return df
    
    def load_txt(self):
        text = self.input.read_text(encoding="utf-8", errors="ignore").splitlines()
        return pd.DataFrame({
            "document_id": self.document_id,
            "paragraph": text
        })
        
    def save_as_parquet(self, output_path = None):
        output_path = output_path or self.input.with_suffix(".parquet")
        self.paragraphs.to_parquet(output_path)
        print(f"Saved extracted document to {output_path}")
    
    def load_parquet(self):
        return pd.read_parquet(self.input)

    def load_document(self):
        suffix = self.input.suffix.lower()

        if suffix == ".pdf":
            return self.load_pdf()
        elif suffix == ".txt":
            return self.load_txt()
        elif suffix == ".parquet":
            return self.load_parquet()
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
    
    def load(self):
        self.paragraphs = self.load_document()

class Corpus:

    """
    Aggregate multiple Loader instances into a single text corpus.

    The Corpus ensures that each Loader has executed its loading step and then
    concatenates all extracted paragraphs into a single pandas DataFrame. This
    provides a unified view over multiple documents for downstream processing
    such as indexing, embedding, or analysis.

    Parameters
    ----------
    *loaders : Loader
        One or more Loader instances. Each loader must expose a `paragraphs`
        attribute after loading, containing a DataFrame with at least
        `document_id` and `paragraph` columns.

    Attributes
    ----------
    paragraphs : pandas.DataFrame
        Combined paragraph-level DataFrame created by concatenating the
        paragraphs from all loaders.

    Notes
    -----
    - If a Loader has not yet been loaded, `Corpus` will call `loader.load()`
      automatically.
    - Paragraphs are concatenated without deduplication or reordering beyond
      the original loader order.
    - All loaders are assumed to produce compatible schemas.
    """

    def __init__(self, *loaders: Loader):
        self.loaders = loaders

        for loader in self.loaders:
            if not hasattr(loader, "paragraphs"):
                loader.load()

        self.combine_paragraphs()
    
    def combine_paragraphs(self):
        self.paragraphs = pd.concat(
            [loader.paragraphs for loader in self.loaders],
            ignore_index=True
        )

class SentenceEmbedder:

    """
    Thin wrapper around a SentenceTransformer for sentence-level embeddings.

    This class provides a minimal abstraction over a SentenceTransformer model,
    exposing convenience methods for embedding single texts or batches of texts.
    It is intended to decouple downstream pipeline components from the concrete
    embedding backend.

    Parameters
    ----------
    sbert : sentence_transformers.SentenceTransformer
        A loaded SentenceTransformer model used to generate embeddings.

    Notes
    -----
    - No preprocessing is applied; texts are passed directly to the model.
    - Output format and dimensionality are determined by the underlying model.
    - `embed_one` is a convenience wrapper and may return a differently shaped
      array than `embed` depending on the SentenceTransformer configuration.
    """

    def __init__(self, sbert: sentence_transformers.SentenceTransformer):
        self.sbert = sbert

    def embed(self, texts: list[str]):
        return self.sbert.encode(texts)

    def embed_one(self, text: str):
        return self.sbert.encode(text)

class SentenceExtractor:

    """
    Extract sentence boundaries and sentence-level spans from paragraphs.

    This class uses a spaCy language pipeline to identify sentence boundaries
    and expose character-level offsets, enabling precise alignment between
    paragraph text and extracted sentences.

    Parameters
    ----------
    spacy : spacy.language.Language
        An initialized spaCy language pipeline with sentence segmentation
        enabled.

    Notes
    -----
    - Sentence boundaries are returned as character offsets relative to the
      input paragraph.
    - No text normalization is performed; offsets correspond exactly to the
      original paragraph string.
    """

    def __init__(self, spacy: spacy.language.Language):
        self.spacy = spacy

    def sentence_boundaries(self, paragraph: str):
        doc = self.spacy(paragraph)
        return list(enumerate(
            [(sent.start_char, sent.end_char) for sent in doc.sents]
        ))

    def extract_sentences(self, paragraph: str, boundaries):
        
        """
        Extract sentence strings from a paragraph using precomputed boundaries.

        Parameters
        ----------
        paragraph : str
            Source paragraph text.
        boundaries : iterable
            Sentence boundary definitions, typically produced by
            `sentence_boundaries()`, containing character start and end offsets.

        Returns
        -------
        list[str]
            Sentences extracted from the paragraph in original order.
        """

        return [paragraph[start:stop] for _, (start, stop) in boundaries]

class Segmenter:
    """
    Segment text into overlapping token windows with character-level boundaries.

    This class wraps a tokenizer that supports offset mappings and produces
    segment boundaries corresponding to sliding token windows. The resulting
    character offsets can be used to slice the original text while maintaining
    alignment with model input segments.

    Parameters
    ----------
    tokenizer
        A tokenizer compatible with Hugging Face tokenizers that supports
        `return_overflowing_tokens` and `return_offsets_mapping`.

    Notes
    -----
    - Segments are defined using the first and last non-special token offsets
      within each window.
    - Overlapping segments are produced using a sliding window controlled by
      `max_length` and `stride`.
    - No text normalization is applied; offsets correspond to the original text.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @staticmethod
    def _segment_boundary(offset_mapping):
        return offset_mapping[1][0], offset_mapping[-2][1]

    def segment_boundaries(
        self,
        text: str,
        **kwargs
    ):
        """
        Compute character-level boundaries for tokenized text segments.

        Parameters
        ----------
        text : str
            Input text to be segmented.
        **kwargs
            Optional tokenizer parameters.
            - max_length : int, optional
                Maximum number of tokens per segment (default: 128).
            - stride : int, optional
                Number of overlapping tokens between consecutive segments
                (default: 32).

        Returns
        -------
        list[tuple[int, int]]
            List of (start_char, end_char) offsets for each segment, relative
            to the original input text.
        """
        
        inputs = self.tokenizer(
            text,
            truncation=True,
            return_overflowing_tokens=True,
            max_length=kwargs.get("max_length", 128),
            stride=kwargs.get("stride", 32),
            return_attention_mask=False,
            return_token_type_ids=False,
            return_offsets_mapping=True
        )

        return [
            self._segment_boundary(mapping)
            for mapping in inputs["offset_mapping"]
        ]

class SegmentAligner:
    """
    Align token-based segments with sentence-level text and embeddings.

    This utility maps precomputed token-window segment boundaries to sentence
    boundaries and selects a non-overlapping set of sentence spans that best
    fit each segment. It optionally propagates sentence-level embeddings to the
    aligned segments.

    Notes
    -----
    - Only sentences fully contained within a segment boundary are considered.
    - When multiple sentence spans map to the same segment, longer spans are
      preferred.
    - Overlapping sentence spans are resolved by greedily keeping the longest
      spans first and dropping duplicates on start and stop indices.
    """

    @staticmethod
    def align(
        segment_boundaries,
        sentence_boundaries,
        sentences,
        embeddings=None
    ):
        """
        Align sentence spans to segment boundaries.

        Parameters
        ----------
        segment_boundaries : list[tuple[int, int]]
            Character-level (start, end) offsets defining token-based segments.
        sentence_boundaries : list[tuple[int, tuple[int, int]]]
            Enumerated sentence boundaries as (sentence_index, (start, end)),
            typically produced by `SentenceExtractor.sentence_boundaries()`.
        sentences : list[str]
            Sentences extracted from the source text, ordered by sentence index.
        embeddings : list or array-like, optional
            Sentence-level embeddings aligned with `sentences`. If provided,
            embeddings are grouped to match the aligned sentence spans.

        Returns
        -------
        sentences_per_segment : list[list[str]]
            Sentences aligned to each selected segment.
        embeddings_per_segment : list[list] or list[None]
            Sentence embeddings grouped per segment, or None if not provided.
        segments : list[str]
            Concatenated sentence text for each segment.
        segment_lengths : list[int]
            Token counts (whitespace-split) for each concatenated segment.
        """
        indices = [
            [
                idx for idx, (start, stop) in sentence_boundaries
                if start >= seg[0] and stop <= seg[1]
            ]
            for seg in segment_boundaries
        ]

        indices = [i for i in indices if i]
        if not indices:
            return [], [], [], []

        df = pd.DataFrame({"indices": indices})
        df["start"] = df.indices.apply(lambda x: x[0])
        df["stop"] = df.indices.apply(lambda x: x[-1])
        df["len"] = df.stop - df.start

        df.sort_values("len", ascending=False, inplace=True)
        for col in ["start", "stop"]:
            df.drop_duplicates(col, keep="first", inplace=True)
        df.sort_index(inplace=True)

        df["sentences"] = df.apply(
            lambda x: sentences[x.start:x.stop + 1], axis=1
        )

        if embeddings is not None:
            df["sentence_embeddings"] = df.apply(
                lambda x: embeddings[x.start:x.stop + 1], axis=1
            )
        else:
            df["sentence_embeddings"] = None

        df["segment"] = df.sentences.apply(" ".join)
        df["segment_length"] = df.segment.str.split().str.len()

        return (
            df.sentences.tolist(),
            df.sentence_embeddings.tolist(),
            df.segment.tolist(),
            df.segment_length.tolist()
        )

class Preprocessor:

    """
    Convert a paragraph-level corpus into aligned, embedded text segments.

    This class orchestrates sentence extraction, optional sentence embedding,
    token-based segmentation, sentence–segment alignment, and final segment
    embedding. The result is a segment-level DataFrame suitable for indexing
    or retrieval tasks.

    Parameters
    ----------
    corpus : Corpus
        Corpus containing paragraph-level text to preprocess.
    embedder : SentenceEmbedder
        Embedder used to generate sentence and segment embeddings.
    sentence_extractor : SentenceExtractor
        Component responsible for sentence boundary detection and extraction.
    segmenter : Segmenter
        Component that defines token-based segment boundaries.

    Attributes
    ----------
    paragraphs : pandas.DataFrame
        Input paragraph-level DataFrame inherited from the corpus.
    segments : pandas.DataFrame
        Segment-level DataFrame produced by `preprocess()`.
    """

    def __init__(
        self,
        corpus: Corpus,
        embedder: SentenceEmbedder,
        sentence_extractor: SentenceExtractor,
        segmenter: Segmenter,
    ):
        self.paragraphs = corpus.paragraphs
        self.embedder = embedder
        self.sentence_extractor = sentence_extractor
        self.segmenter = segmenter

    @staticmethod
    def filter_underlength_paragraphs(df, **kwargs):
        df = df[df.paragraph.str.split().str.len() >= kwargs.get("min_paragraph_length", 32)]
        return df.reset_index(drop=True)

    def preprocess(self, embed_sentences=False, **kwargs):

        """
        Run the full preprocessing pipeline and produce segment-level data.

        The pipeline performs the following steps:
        1. Filter paragraphs shorter than a minimum length.
        2. Compute sentence boundaries and extract sentences.
        3. Optionally embed sentences.
        4. Compute token-based segment boundaries.
        5. Align sentences to segments.
        6. Embed final segment text.
        7. Assign stable segment identifiers per document.

        Parameters
        ----------
        embed_sentences : bool, optional
            Whether to compute and retain sentence-level embeddings
            (default: False).
        **kwargs
            Optional preprocessing parameters:
            - min_paragraph_length : int, optional
                Minimum number of whitespace-separated tokens required for a
                paragraph to be retained (default: 32).
            - max_length : int, optional
                Maximum token length for segments, passed to the Segmenter.
            - stride : int, optional
                Token overlap between consecutive segments.

        Returns
        -------
        pandas.DataFrame
            Segment-level DataFrame containing:
            - document_id
            - segment_id
            - segment
            - segment_embedding
            - sentences
            - sentence_embeddings (if enabled)

        Notes
        -----
        - Progress bars are emitted via `tqdm.pandas`.
        - The method sets `self.segments` as a side effect.
        - Segment identifiers are unique within a document but not globally.
        """

        df = self.paragraphs.copy()

        tqdm.pandas(desc="Filtering paragraphs")
        df = self.filter_underlength_paragraphs(df, **kwargs)

        tqdm.pandas(desc="Sentence boundaries")
        df["sentence_boundaries"] = df.paragraph.progress_apply(
            self.sentence_extractor.sentence_boundaries
        )

        tqdm.pandas(desc="Extracting sentences")
        df["all_sentences"] = df.progress_apply(
            lambda x: self.sentence_extractor.extract_sentences(
                x.paragraph, x.sentence_boundaries
            ),
            axis=1
        )

        if embed_sentences:
            tqdm.pandas(desc="Embedding sentences (batched)")
            df["all_sentence_embeddings"] = df.all_sentences.progress_apply(
                self.embedder.embed
            )
        else:
            df["all_sentence_embeddings"] = None

        tqdm.pandas(desc="Segment boundaries")
        df["segment_boundaries"] = df.paragraph.progress_apply(
            lambda x: self.segmenter.segment_boundaries(x, **kwargs)
        )

        tqdm.pandas(desc="Aligning segments")
        outputs = df.progress_apply(
            lambda x: SegmentAligner.align(
                x.segment_boundaries,
                x.sentence_boundaries,
                x.all_sentences,
                x.all_sentence_embeddings
            ),
            axis=1
        )

        df["sentences"] = outputs.apply(lambda x: x[0])
        df["sentence_embeddings"] = outputs.apply(lambda x: x[1])
        df["segment"] = outputs.apply(lambda x: x[2])
        df["segment_length"] = outputs.apply(lambda x: x[3])

        explode_cols = ["sentences", "segment", "segment_length"]
        if embed_sentences:
            explode_cols.append("sentence_embeddings")

        df = (
            df
            .explode(explode_cols)
            .dropna(subset=["segment"])
            .reset_index(drop=True)
        )


        tqdm.pandas(desc="Embedding segments")
        df["segment_embedding"] = df.segment.progress_apply(
            self.embedder.embed_one
        )

        df["segment_index"] = df.groupby("document_id").cumcount()
        df["segment_id"] = df.apply(
            lambda x: f"{x.document_id}-{x.segment_index:03d}", axis=1
        )

        drop_columns = [
            "paragraph",
            "sentence_boundaries",
            "segment_length",
            "segment_index",
            "all_sentences",
            "all_sentence_embeddings",
            "segment_boundaries"
        ]

        if not embed_sentences:
            drop_columns.append("sentence_embeddings")

        df.drop(columns=drop_columns, inplace=True)
        self.segments = df
        return df
    
class IngestPipeline:

    """
    End-to-end ingestion pipeline for building a FAISS-backed document index.

    This class coordinates preprocessing, optional metadata attachment,
    embedding extraction, FAISS index construction, and persistence of both
    the vector index and its associated document store.

    Parameters
    ----------
    preprocessor : Preprocessor
        Preprocessor instance that produces segment-level text and embeddings.
    metadata : dict, optional
        Mapping from document_id to arbitrary metadata fields. Metadata is
        left-joined onto the segment DataFrame during ingestion.

    Attributes
    ----------
    segments : pandas.DataFrame
        Segment-level records enriched with metadata.
    embeddings : numpy.ndarray
        Array of normalized segment embeddings used to build the FAISS index.
    index : faiss.Index
        FAISS index built over the segment embeddings.
    """

    def __init__(self, preprocessor: Preprocessor, metadata=None):
        self.preprocessor = preprocessor
        self.metadata = metadata
        self.segments = None
        self.embeddings = None
        self.index = None

    def run(self, **kwargs):

        """
        Execute the ingestion pipeline.

        This method ensures preprocessing has been run, validates and merges
        metadata, extracts embeddings, and builds the FAISS index.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments forwarded to the preprocessor.

        Returns
        -------
        IngestPipeline
            The pipeline instance, enabling fluent chaining.
        """

        if not hasattr(self.preprocessor, "segments"):
            self.preprocessor.preprocess(**kwargs)

        self.segments = self.preprocessor.segments
        self._check_metadata()
        self._combine_segments_with_metadata()
        self.embeddings = self._get_embeddings()
        self.index = self._get_index()
        return self

    def _check_metadata(self):
        if not self.metadata:
            return

        segment_ids = set(self.segments["document_id"])
        metadata_ids = set(self.metadata)

        extra = metadata_ids - segment_ids
        missing = segment_ids - metadata_ids

        if extra:
            print("Metadata entries not found in segments:", list(extra))
        if missing:
            print("Segments without metadata:", list(missing))

    def _combine_segments_with_metadata(self):
        if not self.metadata:
            return

        # Convert metadata dict → DataFrame
        metadata_df = (
            pd.DataFrame.from_dict(self.metadata, orient="index")
            .reset_index()
            .rename(columns={"index": "document_id"})
        )

        # Left-join metadata onto segments
        self.segments = self.segments.merge(
            metadata_df,
            on="document_id",
            how="left"
        )


    def _get_embeddings(self):
        embeddings = np.asarray(
            self.segments["segment_embedding"].tolist(),
            dtype="float32"
        )
        return embeddings

    def _get_index(self, index_factory="Flat"):
        embeddings = self.embeddings.copy()
        faiss.normalize_L2(embeddings)

        d = embeddings.shape[1]
        index = faiss.index_factory(
            d,
            index_factory,
            faiss.METRIC_INNER_PRODUCT
        )
        index.add(embeddings)

        return index
   
    def save_index(
        self,
        output_dir: str,
        push_to_hub: bool = False,
        repo_id: str | None = None,
        token: str | None = None,
        repo_type: str = "dataset",
    ):
        """
        Persist the FAISS index and document store to disk and optionally
        upload them to the Hugging Face Hub.

        Two artifacts are written:
        - `index.faiss`: the FAISS vector index
        - `docstore.pkl`: serialized segment metadata and text

        Parameters
        ----------
        output_dir : str
            Directory where index and docstore files will be written.
        push_to_hub : bool, optional
            Whether to upload the generated artifacts to the Hugging Face Hub
            (default: False).
        repo_id : str, optional
            Hugging Face repository ID. Required if `push_to_hub=True`.
        token : str, optional
            Hugging Face authentication token.
        repo_type : str, optional
            Repository type on the Hub (default: "dataset").

        Raises
        ------
        ValueError
            If `push_to_hub=True` and `repo_id` is not provided.

        Notes
        -----
        - Embeddings are not stored in the docstore; only segment text and
          metadata are persisted.
        - FAISS vectors are L2-normalized and indexed using inner product
          similarity.
        - Repositories are created automatically if they do not exist.
        """
        os.makedirs(output_dir, exist_ok=True)

        index_path = os.path.join(output_dir, "index.faiss")
        docstore_path = os.path.join(output_dir, "docstore.pkl")

        # --------------------
        # Write files to disk
        # --------------------
        faiss.write_index(self.index, index_path)
        print(f"Saved index to {index_path}")

        docstore = (
            self.segments
            .drop(columns=["segment_embedding"])
            .to_dict("records")
        )

        with open(docstore_path, "wb") as f:
            pickle.dump(docstore, f)

        print(f"Saved docstore with {len(docstore)} entries to {docstore_path}")
        print(f"Number of vectors in index: {self.index.ntotal}")

        # --------------------
        # Optional: push to Hub
        # --------------------
        if not push_to_hub:
            return

        if repo_id is None:
            raise ValueError("repo_id must be provided when push_to_hub=True")

        api = HfApi(token=token)

        # Ensure repo exists
        try:
            api.repo_info(repo_id=repo_id, repo_type=repo_type)
            print(f"✓ Repo exists: {repo_id}")
        except RepositoryNotFoundError:
            print(f"Repo not found. Creating: {repo_id}")
            api.create_repo(
                repo_id=repo_id,
                repo_type=repo_type,
                exist_ok=True,
            )

        # Upload files (Xet-native)
        upload_file(
            path_or_fileobj=index_path,
            path_in_repo="index.faiss",
            repo_id=repo_id,
            repo_type=repo_type,
            token=token,
            commit_message="Upload FAISS index",
        )

        upload_file(
            path_or_fileobj=docstore_path,
            path_in_repo="docstore.pkl",
            repo_id=repo_id,
            repo_type=repo_type,
            token=token,
            commit_message="Upload docstore",
        )

        print("Successfully pushed index and docstore to Hugging Face Hub")

class Retriever:

    """
    Retrieve and re-rank indexed text segments using bi-encoder and cross-encoder models.

    This class loads a persisted FAISS index and associated document store,
    performs dense retrieval using a sentence-transformer bi-encoder, and
    applies cross-encoder re-ranking to select and highlight the most relevant
    text span for a query.

    Parameters
    ----------
    index_dir : str
        Directory containing `index.faiss` and `docstore.pkl`.
    sbert : sentence_transformers.SentenceTransformer
        Bi-encoder used to embed queries for FAISS retrieval.
    cross_encoder
        Cross-encoder model with a `.predict()` method for re-ranking
        (e.g. SentenceTransformers CrossEncoder).

    Attributes
    ----------
    index : faiss.Index
        FAISS index loaded from disk.
    segments : list[dict]
        Document store entries aligned with FAISS vector order.
    """

    def __init__(self, index_dir, sbert, cross_encoder):
        index, segments = self._load_index(index_dir)
        self.index = index
        self.segments = segments
        self.sbert = sbert
        self.cross = cross_encoder

    def _load_index(self, index_dir):
        index = faiss.read_index(os.path.join(index_dir, "index.faiss"))
        with open(os.path.join(index_dir, "docstore.pkl"), "rb") as f:
            segments = pickle.load(f)
        return index, segments
    
    def preprocess_query(self, query):

        """
        Embed and normalize a query for FAISS search.

        Parameters
        ----------
        query : str
            Input search query.

        Returns
        -------
        numpy.ndarray
            L2-normalized query embedding with shape (1, dim).
        """

        embedding = self.sbert.encode([query]).astype("float32")
        faiss.normalize_L2(embedding)
        return embedding

    def retrieve(self, query, k=50, metadata_fields=None):

        """
        Retrieve the most relevant segment for a query and highlight the best sentence.

        The retrieval process consists of:
        1. Dense retrieval over FAISS using a bi-encoder.
        2. Cross-encoder re-ranking of candidate segments.
        3. Cross-encoder scoring of sentences within the top segment to
           identify a highlight span.

        Parameters
        ----------
        query : str
            User query.
        k : int, optional
            Number of candidate segments retrieved from FAISS (default: 50).
        metadata_fields : list[str], optional
            Metadata fields to include in the result if present in the segment.

        Returns
        -------
        dict
            Retrieval result containing:
            - text : str
                Segment text with the best-matching sentence highlighted.
            - highlight : str
                The selected sentence highlight.
            - <metadata_fields> : optional
                Additional metadata copied from the segment.
        """

        if metadata_fields is None:
            metadata_fields = []

        # ---------- Step 1: SBERT Retrieval ----------
        embedding = self.preprocess_query(query)
        D, I = self.index.search(embedding, k)

        candidates = []
        ce_pairs_segments = []

        for idx in I[0]:
            seg = self.segments[idx]
            candidates.append(seg)
            ce_pairs_segments.append([query, seg["segment"]])

        # ---------- Step 2: Cross-Encoder Re-Rank ----------
        segment_scores = self.cross.predict(ce_pairs_segments)
        best_seg_idx = int(np.argmax(segment_scores))
        best_segment = candidates[best_seg_idx]

        # ---------- Step 3: Sentence Highlighting ----------
        sentences = best_segment["sentences"]
        ce_pairs_sentences = [[query, s] for s in sentences]
        sentence_scores = self.cross.predict(ce_pairs_sentences)

        best_sentence = sentences[int(np.argmax(sentence_scores))].strip()

        highlighted_text = (
            best_segment["segment"]
            .replace(best_sentence, f"**{best_sentence}**")
            .replace("\n", " ")
        )

        result = {
            "text": highlighted_text,
            "highlight": best_sentence
            }

        for field in metadata_fields:
            if field in best_segment:
                result[field] = best_segment[field]

        return result
    
class RAG:

    """
    Retrieval-augmented generation (RAG) pipeline for question answering.

    This class combines a retriever that selects and highlights relevant
    document context with a text-generation model that produces answers
    constrained to that retrieved context.

    Parameters
    ----------
    retriever : Retriever
        Retriever instance responsible for fetching relevant text segments.
    generator : transformers.pipeline
        Text-generation pipeline used to produce answers from retrieved context.

    Notes
    -----
    - The generator is prompted to answer strictly using the provided context.
    - Only the top retrieved segment is used for generation.
    """

    def __init__(
        self,
        retriever : Retriever,
        generator: transformers.pipeline
    ):
        self.retriever = retriever
        self.generator = generator

    
    def answer_query(self, query):

        """
        Answer a question using retrieval-augmented generation.

        The method retrieves a relevant document segment, constructs a
        context-constrained prompt, and generates a response using the
        configured language model.

        Parameters
        ----------
        query : str
            User question.

        Returns
        -------
        dict
            Result containing:
            - response : str
                Generated answer.
            - context : str
                Retrieved text used as context for generation.
            - url : str
                Source URL associated with the retrieved segment.
        """

        doc = self.retriever.retrieve(query, metadata_fields=["url"])

        url = doc["url"]
        context = doc["text"]

        prompt = f"""
        You answer questions strictly using the provided context.

        Context: {context}

        Question: {query}
        """

        out = self.generator(f"<|system|>{prompt}<|assistant|>")[0]["generated_text"]
        response = out.split("<|assistant|>")[-1].strip()

        return{
            "response": response,
            "context": context,
            "url": url
        }
    
class RagDatasetBuilder:

    """
    Build supervised fine-tuning datasets for retrieval-augmented generation.

    This class converts RAG examples into tokenized training instances where
    only the assistant response contributes to the training loss. Prompt and
    context tokens are masked to enforce response-only supervision.

    Parameters
    ----------
    tokenizer
        Tokenizer compatible with the target language model. The tokenizer must
        support special tokens used in the prompt format and provide an EOS
        token.

    Notes
    -----
    - Prompts follow a system / user / assistant chat-style format.
    - Labels are masked (`-100`) for all prompt tokens so that loss is computed
      only on the generated response.
    - Designed for causal language model fine-tuning.
    """
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

        self.system_prompt = """<|system|>
You answer questions strictly using the provided context.
"""

    def build_prompt(self, example):
        return f"""{self.system_prompt}
<|user|>
Context: {example['context']}
Question: {example['query']}
<|assistant|>
""".strip()

    def tokenize_and_mask(self, example):
        prompt = self.build_prompt(example)
        response = example["response"]

        prompt_ids = self.tokenizer(
            prompt,
            add_special_tokens=False
        )["input_ids"]

        response_ids = self.tokenizer(
            response + self.tokenizer.eos_token,
            add_special_tokens=False
        )["input_ids"]

        input_ids = prompt_ids + response_ids

        # Mask everything except the response
        labels = [-100] * len(prompt_ids) + response_ids

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": [1] * len(input_ids),
        }

    def build_dataset(self, dataframe):
        dataset = Dataset.from_pandas(dataframe)

        dataset = dataset.map(
            self.tokenize_and_mask,
            remove_columns=dataset.column_names
        )

        return dataset

    def sanity_check(self, dataset, idx=0):
        ex = dataset[idx]

        input_ids = ex["input_ids"]
        labels = ex["labels"]

        assert len(input_ids) == len(labels), "input_ids / labels length mismatch"

        decoded_input = self.tokenizer.decode(input_ids)

        labeled_token_ids = [
            input_ids[i] for i in range(len(labels)) if labels[i] != -100
        ]
        decoded_labeled = self.tokenizer.decode(labeled_token_ids)

        masked = sum(1 for l in labels if l == -100)
        unmasked = sum(1 for l in labels if l != -100)

        print("=" * 80)
        print("FULL MODEL INPUT (conditioning + target)")
        print("=" * 80)
        print(decoded_input)

        print("\n" + "=" * 80)
        print("TOKENS CONTRIBUTING TO LOSS (response only)")
        print("=" * 80)
        print(decoded_labeled)

        print("\n" + "=" * 80)
        print("MASKING STATS")
        print("=" * 80)
        print(f"Total tokens     : {len(labels)}")
        print(f"Masked tokens    : {masked}")
        print(f"Unmasked tokens  : {unmasked}")
        print("=" * 80)

        assert "<|assistant|>" not in decoded_labeled, \
            "ERROR: Prompt tokens are contributing to loss!"

        print("Sanity check passed: response-only masking is correct.")