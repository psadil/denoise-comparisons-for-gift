library(dplyr)
library(tidyr)
library(purrr)
library(corrr)
library(ggplot2)
# library(mgc)

ukb_ntr <- arrow::open_dataset("derivatives/tc") |>
  filter(
    data == "ukb",
    component == 1,
    method == "abcd",
    ses == 2,
  ) |>
  count(sub, ses, reference) |>
  filter(n > 106) |>
  select(sub, ses, reference) |>
  collect()


ukb <- arrow::open_dataset("derivatives/tc") |>
  semi_join(ukb_ntr) |>
  select(t, component, value, method, sub, reference) |>
  collect() |>
  mutate(
    reference = case_match(
      reference,
      "Neuromark_fMRI_1.0_resampled24" ~ "res-native",
      "Neuromark_fMRI_1.0_resampled" ~ "smoothed"
    )
  ) |>
  pivot_wider(names_from = component) |>
  group_nest(method, sub, reference) |>
  mutate(
    r = map(
      data,
      ~ .x |>
        select(-t) |>
        correlate(method = "spear", quiet = TRUE) |>
        stretch(na.rm = TRUE, remove.dups = TRUE),
      .progress = TRUE
    )
  )

out <- ukb |>
  select(-data) |>
  unnest(r) |>
  mutate(across(c(x, y), as.integer))


out |>
  mutate(r = atanh(r)) |>
  unite(xy, x, y) |>
  pivot_wider(names_from = xy, values_from = r) |>
  group_by(method, reference) |>
  arrow::write_dataset("derivatives/connectivity")

# between(x, 0, 4) ~ "SC",
# between(x, 5, 6) ~ "AU",
# between(x, 7, 15) ~ "SM",
# between(x, 16, 24) ~ "VI",
# between(x, 25, 41) ~ "CC",
# between(x, 42, 48) ~ "DM",
# between(x, 49, 52) ~ "CB"

extra <- function(x, label) {
  annotate("text", x = x, y = -1, label = label, size = 3)
}

out <- duckplyr::read_parquet_duckdb(
  "derivatives/connectivity",
  prudence = "lavish"
) |>
  collect() |>
  summarise(
    across(
      matches("[[:digit:]]+_[[:digit:]]+"),
      ~ tanh(mean(.x, na.rm = TRUE))
    ),
    .by = c(method, reference)
  ) |>
  pivot_longer(c(-method, -reference), values_to = "r") |>
  separate(name, into = c("x", "y"), sep = "_", convert = TRUE)

out |>
  mutate(method = glue::glue("Method: {stringr::str_to_upper(method)}")) |>
  ggplot(aes(x = x, y = y, fill = r)) +
  geom_raster() +
  scico::scale_fill_scico(
    palette = "vik",
    limits = c(-1, 1),
    name = "Avg\nRank\nCor"
  ) +
  geom_hline(yintercept = c(5, 7, 16, 25, 42, 49) - 0.5, alpha = 0.25) +
  geom_vline(xintercept = c(5, 7, 16, 25, 42, 49) - 0.5, alpha = 0.25) +
  facet_grid(reference ~ method) +
  coord_fixed(xlim = c(0, 52), ylim = c(0, 52), clip = "off") +
  extra(0, "SC") +
  extra(5, "AU") +
  extra(10, "SM") +
  extra(19, "VI") +
  extra(28, "CC") +
  extra(45, "DM") +
  extra(52, "CB") +
  xlab("NeuroMark Component") +
  ylab("NeuroMark Component") +
  ggtitle("UKB (N: 1000)") +
  theme_classic(base_size = 18)

ggsave(
  "cormat.png",
  device = ragg::agg_png,
  dpi = 600,
  height = 7.5,
  width = 7.5
)

tmp <- out |>
  mutate(r = atanh(r)) |>
  pivot_wider(names_from = method, values_from = r) |>
  group_nest(x, y, reference) |>
  mutate(
    fit = map(
      data,
      ~ select(.x, -sub) |>
        as.matrix() |>
        irr::icc(model = "t"),
      .progress = TRUE
    ),
    consistency = map_dbl(fit, pluck, "value"),
    ,
    x_network = case_when(
      between(x, 0, 4) ~ "SC",
      between(x, 5, 6) ~ "AU",
      between(x, 7, 15) ~ "SM",
      between(x, 16, 24) ~ "VI",
      between(x, 25, 41) ~ "CC",
      between(x, 42, 48) ~ "DM",
      between(x, 49, 52) ~ "CB"
    ),
    y_network = case_when(
      between(y, 0, 4) ~ "SC",
      between(y, 5, 6) ~ "AU",
      between(y, 7, 15) ~ "SM",
      between(y, 16, 24) ~ "VI",
      between(y, 25, 41) ~ "CC",
      between(y, 42, 48) ~ "DM",
      between(y, 49, 52) ~ "CB"
    ),
    matched_network = x_network == y_network
  ) |>
  select(-data)

tmp |>
  ggplot(aes(x = consistency)) +
  geom_histogram() +
  facet_wrap(~reference, labeller = label_both) +
  xlab("Consistency [ICC(3,1)]")

ggsave("consistency.png", device = ragg::agg_png)


tmp2 <- out |>
  mutate(r = atanh(r)) |>
  unite(xy, x, y) |>
  pivot_wider(names_from = xy, values_from = r) |>
  arrange(reference, sub, method)

ds_native <- discr.stat(
  filter(tmp2, reference == "res-native") |>
    select(-method, -sub, -reference) |>
    as.matrix(),
  filter(tmp2, reference == "res-native") |> select(sub) |> as.matrix()
)

ds_native$discr


ds_smooth <- discr.stat(
  filter(tmp2, reference == "smoothed") |>
    select(-method, -sub, -reference) |>
    as.matrix(),
  filter(tmp2, reference == "smoothed") |> select(sub) |> as.matrix()
)
ds_smooth$discr


Dx <- as.matrix(
  dist(select(tmp2, -method, -sub) |> as.matrix()),
  method = 'euclidian'
)

as_tibble(Dx) |>
  mutate(src = row_number()) |>
  pivot_longer(-src, names_to = "target") |>
  mutate(target = as.integer(target)) |>
  filter(!(target == src)) |>
  # filter(target > src) |>
  ggplot(aes(x = src, y = target, fill = value)) +
  geom_raster() +
  scale_fill_viridis_c(option = "turbo", limits = c(0, NA)) +
  theme_bw() +
  coord_fixed()

ggsave("dist.png", device = ragg::agg_png, width = 7, height = 7, dpi = 900)


arrow::open_dataset("derivatives/predictions/results") |>
  select(starts_with("test"), model, measure, reference, method, fold) |>
  distinct() |>
  collect() |>
  mutate(test_r2 = if_else(test_r2 < 0, 0, test_r2)) |>
  pivot_longer(starts_with("test")) |>
  mutate(
    method = stringr::str_extract(method, "(abcd)|(hcp)"),
    reference = case_match(
      reference,
      "'res-native'" ~ "res-native",
      .default = "3mm-smooth"
    ),
    name = stringr::str_remove(name, "test_")
  ) |>
  ggplot(aes(x = method, y = value, color = reference)) +
  facet_wrap(~name, scales = "free_y") +
  geom_boxplot(outliers = FALSE) +
  geom_point(position = position_jitterdodge(), alpha = 0.2)

ggsave("fluid-intelligence-predictions.png", device = ragg::agg_png, width = 6)
