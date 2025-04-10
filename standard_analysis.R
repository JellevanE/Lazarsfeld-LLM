library(tidyverse)

# Read the data
df_long <- read_csv("evaluation_results/model_score_comparison_flat.csv")

# Pivot wider so each combination of (label, dimension, question) is one row,
# and each model is a new column with its corresponding score.
df_wide <- df_long %>%
  filter(!is.na(dimension)) %>%
  pivot_wider(
    id_cols     = c("label", "dimension", "question"),  # group by these
    names_from  = "model",                              # each model gets a column
    values_from = "score"                               # fill with that model’s score
  )

df_wide_mean <- df_wide %>%
  rowwise() %>%
  mutate(aggregate_models = mean(c_across(c("gpt-4-turbo", "gpt-3.5-turbo-0125", "gpt-4o")), na.rm = TRUE)) %>%
  ungroup()


# Identify reference column
ref_col <- "LL-01-pro"

# Collect the other model columns to compare
model_cols <- setdiff(colnames(df_wide_mean), c("label", "dimension", "question", ref_col))

# Compute Spearman correlations vs. LL-01-pro
spearman_results <- map_dfr(model_cols, function(m) {
  cor_val <- cor(
    df_wide_mean[[ref_col]],
    df_wide_mean[[m]],
    method = "spearman",
    use = "complete.obs"  # ignore NA rows
  )
  tibble(
    model      = m,
    spearman_r = cor_val
  )
})


df_long %>%
  filter(!is.na(dimension)) %>%
  group_by(dimension) %>%
  group_modify(~{
    wide <- pivot_wider(.x,
                        id_cols = c("label", "question"),
                        names_from = model,
                        values_from = score)
    other_models <- setdiff(colnames(wide), c("label", "question", "LL-01-pro"))
    map_dfr(other_models, function(m) {
      if (all(is.na(wide[[m]]))) return(NULL)
      tibble(
        model = m,
        spearman_r = cor(wide[["LL-01-pro"]], wide[[m]], method = "spearman", use = "complete.obs")
      )
    }) %>% mutate(dimension = unique(.x$dimension))
  }) %>% 
  relocate(dimension, model, spearman_r)

# Compute absolute errors per model
df_diff <- df_wide_mean %>%
  mutate(across(
    all_of(model_cols),
    ~ abs(. - .data[[ref_col]]),
    .names = "diff_{.col}"
  ))

# Compute summary deviation per row (e.g., max or mean of all model differences)
df_outlier_scores <- df_diff %>%
  rowwise() %>%
  mutate(
    max_diff = max(c_across(starts_with("diff_")), na.rm = TRUE),
    mean_diff = mean(c_across(starts_with("diff_")), na.rm = TRUE),
    avg_diff_excl_3.5 = mean(
      c_across(starts_with("diff_") & !contains("gpt-3.5-turbo-0125")
      ),
      na.rm = TRUE
    )
  ) %>%
  ungroup()

# Sort to find most "outlier-ish" questions
outliers_by_max <- df_outlier_scores %>%
  arrange(desc(max_diff)) %>%
  select(label, dimension, question, starts_with("diff_"), max_diff, mean_diff)

# Calculate avg dif per question to determine outlier questions
avg_diffs_by_question_dim <- df_outlier_scores %>%
  group_by(dimension) %>%
  summarise(across(
    c(starts_with("diff_"), mean_diff, max_diff, avg_diff_excl_3.5),
    ~ mean(.x, na.rm = TRUE),
    .names = "avg_{.col}"
  ), .groups = "drop")

a = sum(avg_diffs_by_question_dim$avg_avg_diff_excl_3.5)
b = sum(avg_diffs_by_question_dim$avg_mean_diff)
c = (a, b)



