# ============================================================
# IDS 570: Text as Data — Data Exploration
# Political Economies Corpus (Malynes / Misselden, c.1601–1623)
# Author: Supriya Nannapaneni (sn367)
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
raw_texts <- str_replace_all(raw_texts, "\u017f", "s")   # ſ → s
names(raw_texts) <- doc_ids

# Normalizations

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
  "therewith", "herewith", "whatsoever", "unto",
  "saith", "sayeth", "maketh", "taketh", "hast"
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
  # Remove under 4 characters or starting with a digit
  tokens_remove(pattern = "^.{1,3}$", valuetype = "regex") |>
  tokens_remove(pattern = "^[0-9]",   valuetype = "regex")

# Construct DFM (raw counts)
dfm_counts <- dfm(toks)
cat("\nDFM dimensions (documents x features):", dim(dfm_counts), "\n")

# TF-IDF weights
dfm_tfidf <- dfm_tfidf(dfm_counts)

# Extract top N terms per document 
n_top_terms <- 12   # <-- change here if you want more or fewer

tfidf_long <- convert(dfm_tfidf, to = "data.frame") |>
  pivot_longer(-doc_id, names_to = "term", values_to = "tfidf") |>
  filter(tfidf > 0) |>
  group_by(doc_id) |>
  slice_max(order_by = tfidf, n = n_top_terms) |>
  arrange(doc_id, desc(tfidf)) |>
  ungroup()

cat("\n=== TOP", n_top_terms, "TF-IDF TERMS BY DOCUMENT ===\n")
print(tfidf_long, n = Inf)

# One-row-per-document summary (useful for the report) 
tfidf_summary <- tfidf_long |>
  mutate(tfidf = round(tfidf, 4)) |>
  group_by(doc_id) |>
  summarize(top_terms = paste(term, collapse = ", "), .groups = "drop")

cat("\n=== TF-IDF SUMMARY TABLE ===\n")
print(tfidf_summary)

# Visualize: faceted bar chart of top terms per document 
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

cat("\n--- Top 5 most similar pairs ---\n")
print(slice_max(r_pairs, order_by = r, n = 5))

cat("\n--- Top 5 least similar pairs ---\n")
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

# APPROACH 3 — Syntactic Complexity (udpipe)

# We analyze only TWO texts, chosen from the Pearson / TF-IDF results.
#
# The two documents are selected automatically below:
#   - syntax_doc_1: the most isolated document (lowest average r with others)
#   - syntax_doc_2: the most central document (highest average r with others)

# Average correlation with all other documents (excluding self)
avg_r <- rowMeans(r_mat - diag(nrow(r_mat))) / (ncol(r_mat) - 1)

syntax_doc_1 <- names(which.min(avg_r))   # most isolated
syntax_doc_2 <- names(which.max(avg_r))   # most central

cat("\nSyntactic complexity analysis:\n")
cat("  Doc 1 (most isolated): ", syntax_doc_1, "\n")
cat("  Doc 2 (most central):  ", syntax_doc_2, "\n")

# Load whichever udpipe model file is in the working directory
model_path <- list.files(".", pattern = "english.*\\.udpipe", full.names = TRUE)[1]
if (is.na(model_path)) {
  stop("No udpipe model found in working directory. Run udpipe_download_model('english') first.")
}
model <- udpipe_load_model(model_path)

# Cap input length so parsing is tractable on any machine
max_words <- 30000

prep_text <- function(text, doc_name) {
  words <- str_split(text, "\\s+")[[1]]
  if (length(words) > max_words) {
    message("  Truncating '", doc_name, "' to ", max_words, " words for parsing.")
    words <- words[seq_len(max_words)]
  }
  paste(words, collapse = " ")
}

annotate_doc <- function(raw_text, doc_name) {
  prepped <- prep_text(raw_text, doc_name)
  ann     <- udpipe_annotate(model, x = prepped, doc_id = doc_name)
  as.data.frame(ann)
}

cat("Parsing documents — this may take several minutes...\n")
ann1 <- annotate_doc(raw_texts[syntax_doc_1], syntax_doc_1)
ann2 <- annotate_doc(raw_texts[syntax_doc_2], syntax_doc_2)

# Compute all required syntactic measures 
compute_syntax_measures <- function(ann_df, doc_label) {
  
  all_sent_ids <- unique(ann_df$sentence_id)
  n_sents      <- length(all_sent_ids)
  
  # MLS: mean content tokens per sentence (punctuation excluded)
  mls <- ann_df |>
    filter(upos != "PUNCT") |>
    group_by(sentence_id) |>
    summarize(n = n(), .groups = "drop") |>
    pull(n) |>
    mean()
  
  # Dependent clauses per sentence: count SCONJ tokens as subordinate clause markers
  sconj_df <- ann_df |>
    filter(upos == "SCONJ") |>
    count(sentence_id, name = "n_sconj")
  
  dc_df <- tibble(sentence_id = all_sent_ids) |>
    left_join(sconj_df, by = "sentence_id") |>
    mutate(n_sconj = replace_na(n_sconj, 0))
  
  dc_per_sent <- mean(dc_df$n_sconj)
  
  # Clauses per sentence: main clause + one per subordinating conjunction
  cs <- mean(1 + dc_df$n_sconj)
  
  # Coordination per sentence: count CCONJ tokens
  cconj_df <- ann_df |>
    filter(upos == "CCONJ") |>
    count(sentence_id, name = "n_cconj")
  
  coord_df <- tibble(sentence_id = all_sent_ids) |>
    left_join(cconj_df, by = "sentence_id") |>
    mutate(n_cconj = replace_na(n_cconj, 0))
  
  coord_per_sent <- mean(coord_df$n_cconj)
  
  # Complex nominals per sentence: dependency relations that modify/expand NPs
  cn_rels <- c("amod", "nmod", "nummod", "appos", "acl", "relcl")
  
  cn_df_raw <- ann_df |>
    filter(dep_rel %in% cn_rels) |>
    count(sentence_id, name = "n_cn")
  
  cn_df <- tibble(sentence_id = all_sent_ids) |>
    left_join(cn_df_raw, by = "sentence_id") |>
    mutate(n_cn = replace_na(n_cn, 0))
  
  cn_per_sent <- mean(cn_df$n_cn)
  
  tibble(
    Document   = doc_label,
    Sentences  = n_sents,
    MLS        = round(mls, 1),
    `C/S`      = round(cs, 2),
    `DC/S`     = round(dc_per_sent, 2),
    `Coord/S`  = round(coord_per_sent, 2),
    `CN/S`     = round(cn_per_sent, 2)
  )
}

syntax1 <- compute_syntax_measures(ann1, syntax_doc_1)
syntax2 <- compute_syntax_measures(ann2, syntax_doc_2)

syntax_table <- bind_rows(syntax1, syntax2)

cat("\n=== SYNTACTIC COMPLEXITY TABLE ===\n")
print(syntax_table)

# Pull a representative example sentence per document 
# Selects the sentence with the most subordinating conjunctions
# within a readable token-length window

get_example_sentence <- function(ann_df, doc_label,
                                 min_tokens = 30, max_tokens = 90) {
  example <- ann_df |>
    group_by(sentence_id) |>
    summarize(
      n_tokens = sum(upos != "PUNCT"),
      n_sconj  = sum(upos == "SCONJ"),
      text     = first(sentence),
      .groups  = "drop"
    ) |>
    filter(n_tokens >= min_tokens, n_tokens <= max_tokens) |>
    slice_max(order_by = n_sconj, n = 1, with_ties = FALSE)
  
  if (nrow(example) == 0) {
    message("No example sentence found in [", min_tokens, ", ", max_tokens,
            "] token range for: ", doc_label)
    return(invisible(NULL))
  }
  
  cat("\n--- Example sentence:", doc_label, "---\n")
  cat(example$text[1], "\n")
  cat("[tokens:", example$n_tokens[1],
      "| subordinating conjunctions:", example$n_sconj[1], "]\n")
}

get_example_sentence(ann1, syntax_doc_1)
get_example_sentence(ann2, syntax_doc_2)

# Bar chart comparing the syntactic measures 
p_syntax <- syntax_table |>
  pivot_longer(-c(Document, Sentences), names_to = "Measure", values_to = "Value") |>
  ggplot(aes(x = Document, y = Value, fill = Document)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ Measure, scales = "free_y") +
  labs(
    title    = "Syntactic Complexity Comparison",
    subtitle = paste("First", max_words, "words of each document; udpipe English EWT model"),
    x = NULL, y = NULL
  ) +
  theme_minimal(base_size = 11) +
  theme(
    axis.text.x = element_text(angle = 30, hjust = 1),
    plot.title  = element_text(face = "bold")
  )

ggsave("syntax_comparison.png", plot = p_syntax, width = 9, height = 6, dpi = 150)
cat("Syntax comparison chart saved to syntax_comparison.png\n")


# FINAL SUMMARY PRINT-OUT
cat("\n\n KEY NUMBERS FOR REPORT \n")

cat("\nMost similar pair:\n")
print(slice_max(r_pairs, order_by = r, n = 1))

cat("\nLeast similar pair:\n")
print(slice_min(r_pairs, order_by = r, n = 1))

cat("\nSyntactic complexity table:\n")
print(syntax_table)

cat("\nTF-IDF summary (one row per document):\n")
print(tfidf_summary)