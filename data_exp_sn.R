# ============================================================
# IDS 570: Text as Data — Data Exploration
# Author: Supriya Nannapaneni
# Political Economies Corpus (Malynes / Misselden, c.1601–1623)
# ============================================================

library(dplyr)
library(tidyr)
library(tidytext)
library(tibble)
library(readr)
library(tidyverse)
library(purrr)
library(quanteda)
library(quanteda.textstats)
library(stringr)

# STEP 0 — Load texts and normalize

all_files <- list.files("all_text_files/", full.names = TRUE)
all_files

# Load and name — use filename without extension as document ID
raw_texts  <- sapply(all_files, function(f) paste(readLines(f, warn = FALSE), collapse = " "))
doc_ids    <- tools::file_path_sans_ext(basename(all_files))  # save names separately
names(raw_texts) <- doc_ids

cat("Files loaded:", length(raw_texts), "\n")
cat("Document names:\n")
print(doc_ids)

# Required: normalize long S
# str_replace_all() drops names from a named vector, so we restore them after
# each normalization step using the saved doc_ids vector.
raw_texts <- str_replace_all(raw_texts, "\u017f", "s")   # ſ → s
names(raw_texts) <- doc_ids

# Optional normalizations (with rationale)

# Remove non-Latin alphabet placeholder tags that appear in EEBO transcriptions.
# These represent Greek/Hebrew citations rendered as 〈 in non-Latin alphabet 〉
# and would produce meaningless tokens in the DFM.
raw_texts <- str_replace_all(raw_texts, "\u3008[^\u3009]*\u3009", "")
names(raw_texts) <- doc_ids

# Remove typesetting artefacts (bullet/dagger marks used as marginal indicators)
raw_texts <- str_replace_all(raw_texts, "[●▪◊]", "")
names(raw_texts) <- doc_ids

# We deliberately do NOT normalize u/v or i/j variation, and we do NOT
# collapse variant spellings (e.g. mony/money, moneys/monies). These
# differences are preserved because they may reflect genuine authorial,
# temporal, or genre-level distinctions worth keeping visible in the analysis.

cat("\nWord counts after normalization:\n")
print(sapply(str_split(raw_texts, "\\s+"), length))

# Early Modern stopwords — added on top of the quanteda default
# These are grammatical/functional words not in the standard English list

em_stopwords <- c(
  "hath", "doth", "thee", "thou", "hee", "wee", "doe", "vpon", "vnto",
  "haue", "bee", "yt", "ye", "thence", "hence", "hither", "thither",
  "thereof", "therein", "thereto", "thereupon", "whereby", "wherein",
  "hereof", "herein", "hereto", "whereof", "whereto", "whereunto",
  "therewith", "herewith","saith", "sayeth", "maketh", "taketh", "hast"
)


# APPROACH 1 — TF–IDF: Lexical Distinctiveness

# Build corpus
corp <- corpus(
  data.frame(
    doc_id = names(raw_texts),
    text   = unname(raw_texts),
    stringsAsFactors = FALSE
  ),
  docid_field = "doc_id",
  text_field  = "text"
)

# Known artefact tokens to remove explicitly (OCR noise, abbreviation fragments,
# typographic units that survive punctuation removal)
artefact_tokens <- c(
  # abbreviation fragments from accounting tables and marginal refs
  "ll", "ss", "arg", "answ", "starlin", "per", "cent",
  # ordinal/Latin abbreviations
  "dly", "ndly", "rdly", "thly", "viz", "ibid", "idem",
  # single stray letters that survive as tokens
  letters
)

# Tokenize
toks <- tokens(
  corp,
  remove_punct    = TRUE,
  remove_numbers  = TRUE,
  remove_symbols  = TRUE
) |>
  tokens_tolower() |>
  tokens_remove(c(stopwords("en"), em_stopwords, artefact_tokens)) |>
  # Remove anything under 4 characters or starting with a digit
  tokens_remove(pattern = "^.{1,3}$", valuetype = "regex") |>
  tokens_remove(pattern = "^[0-9]",   valuetype = "regex")

# Construct DFM (raw counts)
dfm_counts <- dfm(toks)
cat("\nDFM dimensions (documents x features):", dim(dfm_counts), "\n")

# TF-IDF weights
dfm_tfidf <- dfm_tfidf(dfm_counts)

# Extract top N terms per document
n_top_terms <- 12   

tfidf_long <- convert(dfm_tfidf, to = "data.frame") |>
  pivot_longer(-doc_id, names_to = "term", values_to = "tfidf") |>
  filter(tfidf > 0) |>
  group_by(doc_id) |>
  slice_max(order_by = tfidf, n = n_top_terms) |>
  arrange(doc_id, desc(tfidf)) |>
  ungroup()

cat("\n TOP", n_top_terms, "TF-IDF TERMS BY DOCUMENT \n")
print(tfidf_long, n = Inf)

# One-row-per-document summary 
tfidf_summary <- tfidf_long |>
  mutate(tfidf = round(tfidf, 4)) |>
  group_by(doc_id) |>
  summarize(top_terms = paste(term, collapse = ", "), .groups = "drop")

cat("\n=== TF-IDF SUMMARY TABLE ===\n")
print(tfidf_summary)

# Visualize: faceted bar chart of top terms per document ----
# scales = "free" gives each panel its own x AND y axis, preventing one outlier
# document from compressing all the others. Terms are ranked within each panel.

p_tfidf <- tfidf_long |>
  mutate(term = reorder_within(term, tfidf, doc_id)) |>
  ggplot(aes(x = tfidf, y = term, fill = doc_id)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ doc_id, scales = "free", ncol = 4) +
  scale_y_reordered() +
  labs(
    title    = "Top TF-IDF Terms by Document",
    subtitle = paste("Top", n_top_terms, "terms; long S normalized; EM stopwords removed"),
    x        = "TF-IDF weight",
    y        = NULL
  ) +
  theme_minimal(base_size = 9) +
  theme(
    plot.title      = element_text(face = "bold", size = 12),
    strip.text      = element_text(face = "bold", size = 8),
    axis.text.y     = element_text(size = 7),
    axis.text.x     = element_text(size = 7),
    panel.spacing   = unit(0.8, "lines")
  )

# Taller output so each panel has room for its terms
n_docs <- n_distinct(tfidf_long$doc_id)
plot_height <- ceiling(n_docs / 4) * 2.8

ggsave("tfidf_barchart.png", plot = p_tfidf,
       width = 14, height = plot_height, dpi = 150, limitsize = FALSE)
cat("TF-IDF bar chart saved to tfidf_barchart.png\n")

# APPROACH 2 — Pearson Correlation: Similarity / Distance

# Trim very rare words before computing correlations.
# Words appearing fewer than min_freq times across the corpus are removed.
# This prevents rare artefact tokens from distorting the correlation.
min_freq <- 5  

dfm_trimmed <- dfm_trim(dfm_counts, min_termfreq = min_freq)
cat("\nTrimmed DFM dimensions (min_termfreq =", min_freq, "):", dim(dfm_trimmed), "\n")

# Pairwise Pearson correlations between all documents
sim_r <- textstat_simil(dfm_trimmed, margin = "documents", method = "correlation")

# Convert to matrix and round
r_mat <- as.matrix(sim_r)
r_mat <- round(r_mat, 3)

cat("\n=== PEARSON CORRELATION MATRIX ===\n")
print(r_mat)

# Most / least similar pairs (excluding self-comparisons) 
r_pairs <- as.data.frame(r_mat) |>
  rownames_to_column("doc_i") |>
  pivot_longer(-doc_i, names_to = "doc_j", values_to = "r") |>
  filter(doc_i < doc_j)   # each pair appears once

cat("\n Top 5 most similar pairs\n")
print(slice_max(r_pairs, order_by = r, n = 5))

cat("\nTop 5 least similar pairs\n")
print(slice_min(r_pairs, order_by = r, n = 5))

# Heatmap
# Shorten labels so they fit on the plot axes
label_max_chars <- 10   

short_names     <- str_trunc(rownames(r_mat), label_max_chars, ellipsis = "")
r_mat_plot      <- r_mat
rownames(r_mat_plot) <- short_names
colnames(r_mat_plot) <- short_names

heat_df <- as.data.frame(r_mat_plot) |>
  rownames_to_column("doc_i") |>
  pivot_longer(-doc_i, names_to = "doc_j", values_to = "r")

heat_df$doc_i <- factor(heat_df$doc_i, levels = short_names)
heat_df$doc_j <- factor(heat_df$doc_j, levels = short_names)

p_heat <- ggplot(heat_df, aes(x = doc_j, y = doc_i, fill = r)) +
  geom_tile(color = "white") +
  geom_text(aes(label = sprintf("%.2f", r)), size = 3, color = "black") +
  coord_fixed() +
  scale_fill_gradient2(
    low      = "steelblue",
    mid      = "white",
    high     = "firebrick",
    midpoint = 0,
    limits   = c(-1, 1),
    name     = "Pearson r"
  ) +
  labs(
    title    = "Pairwise Pearson Correlation Between Documents",
    subtitle = paste("DFM trimmed to min_termfreq =", min_freq),
    x = NULL,
    y = NULL
  ) +
  theme_minimal(base_size = 11) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    panel.grid  = element_blank(),
    plot.title  = element_text(face = "bold")
  )

ggsave("pearson_heatmap.png", plot = p_heat, width = 7, height = 6, dpi = 150)
cat("Heatmap saved to pearson_heatmap.png\n")

# APPROACH 3 — Surface-Level Register Complexity

# We use parser-free surface measures that do not depend on a dependency
# parser trained on modern English, making them appropriate for early modern
# texts where parser accuracy cannot be assumed.
#
# Five measures are computed for ALL documents:
#   MSL    — Mean Sentence Length (tokens per sentence)
#   MWL    — Mean Word Length (characters per token; proxy for Latinate register)
#   MATTR  — Moving-Average Type-Token Ratio (window = mattr_window tokens)
#             Size-independent lexical diversity measure
#   PD     — Punctuation Density (punctuation marks per 100 tokens)
#             Proxy for syntactic segmentation density
#   LWR    — Long-Word Ratio (proportion of tokens >= long_word_chars characters)
#             Length-independent proxy for technical/Latinate vocabulary

mattr_window    <- 100   # <-- MATTR sliding window size
long_word_chars <- 7     # <-- threshold for "long word" in LWR

# Helper: split raw text into sentences (period/!/?-delimited)

split_sentences <- function(text) {
  # Split on sentence-ending punctuation followed by whitespace or end of string.
  # This is intentionally simple: we want consistent behaviour across all docs
  # without any language-model assumptions.
  sents <- str_split(text, "(?<=[.!?])\\s+")[[1]]
  sents <- str_trim(sents)
  sents[nchar(sents) > 0]
}

# Helper: split sentence into word tokens (alpha only, no punct)
word_tokens <- function(text) {
  str_extract_all(text, "[A-Za-z]+")[[1]]
}

# Helper: MATTR
compute_mattr <- function(tokens, window) {
  n <- length(tokens)
  if (n < window) return(length(unique(tokens)) / n)  # fall back for very short docs
  ttrs <- numeric(n - window + 1)
  for (i in seq_along(ttrs)) {
    w <- tokens[i:(i + window - 1)]
    ttrs[i] <- length(unique(w)) / window
  }
  mean(ttrs)
}

# Compute all measures for one document
compute_surface_measures <- function(text, doc_label) {
  
  # Sentences
  sents     <- split_sentences(text)
  n_sents   <- length(sents)
  
  # All word tokens from the full text
  all_words <- word_tokens(text)
  n_words   <- length(all_words)
  
  # MSL: mean tokens per sentence
  sent_lengths <- sapply(sents, function(s) length(word_tokens(s)))
  msl <- mean(sent_lengths[sent_lengths > 0])
  
  # MWL: mean characters per word token
  mwl <- mean(nchar(all_words))
  
  # MATTR
  mattr <- compute_mattr(tolower(all_words), mattr_window)
  
  # PD: punctuation marks per 100 tokens
  # Count punctuation characters in the raw text
  n_punct <- nchar(gsub("[^.!?,;:()\\[\\]\\-\"\']", "", text))
  pd <- (n_punct / n_words) * 100
  
  # LWR: proportion of tokens >= long_word_chars characters
  lwr <- mean(nchar(all_words) >= long_word_chars)
  
  tibble(
    Document = doc_label,
    Tokens   = n_words,
    MSL      = round(msl, 1),
    MWL      = round(mwl, 2),
    MATTR    = round(mattr, 3),
    PD       = round(pd, 1),
    LWR      = round(lwr, 3)
  )
}

cat("\nComputing surface measures for all documents...\n")

surface_table <- map2_dfr(
  raw_texts, names(raw_texts),
  compute_surface_measures
)

cat("\n=== SURFACE REGISTER MEASURES ===\n")
print(surface_table, n = Inf)

# Visualise: faceted dot plot, all documents, all measures
p_surface <- surface_table |>
  pivot_longer(-c(Document, Tokens), names_to = "Measure", values_to = "Value") |>
  mutate(Document = fct_reorder(Document, Value, .fun = mean)) |>
  ggplot(aes(x = Value, y = Document, color = Measure)) +
  geom_point(size = 2.5, show.legend = FALSE) +
  geom_segment(aes(x = 0, xend = Value, yend = Document),
               linewidth = 0.4, show.legend = FALSE) +
  facet_wrap(~ Measure, scales = "free_x", ncol = 3) +
  labs(
    title    = "Surface Register Complexity — All Documents",
    subtitle = paste0("MSL: mean sentence length; MWL: mean word length; ",
                      "MATTR: moving-avg TTR (window=", mattr_window, "); ",
                      "PD: punct/100 tokens; LWR: prop. tokens >= ", long_word_chars, " chars"),
    x = NULL, y = NULL
  ) +
  theme_minimal(base_size = 9) +
  theme(
    plot.title    = element_text(face = "bold", size = 11),
    strip.text    = element_text(face = "bold", size = 9),
    axis.text.y   = element_text(size = 7),
    panel.spacing = unit(0.8, "lines")
  )

n_docs_surf  <- nrow(surface_table)
surf_height  <- ceiling(n_docs_surf / 5) * 3.5 + 2

ggsave("surface_measures.png", plot = p_surface,
       width = 12, height = surf_height, dpi = 150, limitsize = FALSE)
cat("Surface measures plot saved to surface_measures.png\n")


# FINAL SUMMARY PRINT-OUT


cat("\n\n========== KEY NUMBERS ==========\n")

cat("\nMost similar pair:\n")
print(slice_max(r_pairs, order_by = r, n = 1))

cat("\nLeast similar pair:\n")
print(slice_min(r_pairs, order_by = r, n = 1))

cat("\nSurface register measures table:\n")
print(surface_table, n = Inf)

cat("\nTF-IDF summary (one row per document):\n")
print(tfidf_summary)
