import altair as alt
import polars as pl
import polars.selectors as cs
import streamlit as st
from kedro.config import OmegaConfigLoader
from kedro.io.data_catalog import DataCatalog

from vn1_sales_forecast.expr import competition_score
from vn1_sales_forecast.settings import PRED_PREFIX

st.set_page_config(layout="wide")

st.title("Model Contest")


@st.cache_resource
def create_catalog() -> DataCatalog:
    loader = OmegaConfigLoader(conf_source="conf")
    catalog = DataCatalog.from_config(loader.get("catalog"))
    return catalog


catalog = create_catalog()

total_errors: pl.LazyFrame = catalog.load("total_errors")
total_scores: pl.LazyFrame = catalog.load("total_scores")
cv_scores: pl.LazyFrame = catalog.load("cv_scores")
cv_forecast: pl.LazyFrame = catalog.load("cv_forecast")
sales: pl.LazyFrame = catalog.load("primary_sales")
scale_category: pl.LazyFrame = catalog.load("scale_category_analysis")
winner_model: pl.LazyFrame = catalog.load("winner_model_data")
horizon_errors: pl.LazyFrame = catalog.load("horizon_errors")

model_sort = total_scores.sort("score").select("model").collect().to_series().to_list()


# Model
models: list[str] = cv_forecast.select(cs.starts_with(PRED_PREFIX)).collect_schema().names()
models = [m.removeprefix(PRED_PREFIX) for m in models]

model_group = st.selectbox("Models Group", ["Single", "Ensemble", "All"])
if model_group == "Single":
    models = [m for m in models if "Ensemble" not in m]
elif model_group == "Ensemble":
    models = [m for m in models if "Ensemble" in m]


selected_models = st.multiselect("Models", models, default=sorted(set(models) - {"ZeroModel"}))
selected_model_cols = [f"{PRED_PREFIX}{m}" for m in selected_models]

total_errors = total_errors.filter(pl.col("model").is_in(selected_models))
total_scores = total_scores.filter(pl.col("model").is_in(selected_models))
horizon_errors = horizon_errors.filter(pl.col("model").is_in(selected_models))
cv_scores = cv_scores.select(cs.exclude(cs.starts_with(PRED_PREFIX)), *selected_model_cols)
cv_forecast = cv_forecast.select(cs.exclude(cs.starts_with(PRED_PREFIX)), *selected_model_cols)
winner_model = winner_model.filter(pl.col("model").is_in(selected_models))

# Leaderboard -------------------------------------------------------------------------------------
st.subheader("Leaderboard")

with st.expander("Table"):
    st.dataframe(
        total_scores.sort("score").with_columns((pl.arange(pl.len()) + 1).alias("rank")).collect(),
        hide_index=True,
        use_container_width=True,
    )

# Plot 1 -------------------------------------------------------------------------------------
st.subheader("Average Scores by Model")
c = (
    alt.Chart(total_scores.collect())
    .mark_bar()
    .encode(
        x=alt.X("model", sort=model_sort),
        y="score",
        color=alt.Color("model", sort=model_sort),
        tooltip=["model", "score"],
    )
)
st.altair_chart(c, use_container_width=True)  # type: ignore

# Plot 2 -------------------------------------------------------------------------------------
st.subheader("CV Scores")
group_by_model = st.checkbox("Group by model", value=True)
cutoffs = (
    cv_scores.select(pl.col("cutoff_date").cast(pl.String).unique().sort()).collect().to_series()
)
selected_cutoffs = st.multiselect("Cutoff Dates", cutoffs, default=cutoffs)
d = (
    cv_scores.filter(pl.col("cutoff_date").cast(pl.String).is_in(selected_cutoffs))
    .with_columns(pl.col("cutoff_date").dt.to_string(r"%Y-%m-%d"))
    .unpivot(index="cutoff_date", variable_name="model", value_name="score")
    .with_columns(pl.col("model").str.strip_prefix(PRED_PREFIX))
    .collect()
)

cv_model_sort = d.group_by("model").agg(pl.col("score").mean()).sort("score")["model"].to_list()

c = (
    alt.Chart(d)
    .mark_bar()
    .encode(
        y="score",
        color=alt.Color("model", sort=model_sort),
        tooltip=["model", "score", "cutoff_date"],
    )
)

if group_by_model:
    c = c.encode(
        x=alt.X("model", sort=cv_model_sort),
        opacity="cutoff_date:N",
        xOffset="cutoff_date:N",
    )
else:
    c = c.encode(
        x="cutoff_date",
        xOffset=alt.XOffset("model", sort=cv_model_sort),
    )

st.altair_chart(c, use_container_width=True)  # type: ignore

# Plot 2.1 -------------------------------------------------------------------------------------
st.subheader("CV Rank")

cv_ranks = (
    cv_scores.unpivot(index="cutoff_date", variable_name="model", value_name="score")
    .with_columns(pl.col("model").str.strip_prefix(PRED_PREFIX))
    .with_columns(pl.col("score").rank("ordinal").over("cutoff_date").alias("rank"))
)

selection = alt.selection_point(fields=["model"], bind="legend")

c = (
    alt.Chart(cv_ranks.collect())
    .mark_line(point=True)
    .encode(
        x="cutoff_date",
        y=alt.Y("rank", scale=alt.Scale(reverse=True)),
        color=alt.Color("model", sort=model_sort),
        tooltip=["model", "rank", "score"],
        opacity=alt.condition(selection, alt.value(1), alt.value(0.07)),
    )
    .add_params(selection)
)
st.altair_chart(c, use_container_width=True)  # type: ignore

mean_rank = cv_ranks.group_by("model").agg(pl.mean("rank").alias("mean_rank"))
rank_model_sort = mean_rank.sort("mean_rank").select("model").collect().to_series().to_list()

c = (
    alt.Chart(mean_rank.collect())
    .mark_bar()
    .encode(
        x=alt.X("model", sort=rank_model_sort),
        y="mean_rank",
        color=alt.Color("model", sort=model_sort),
        tooltip=["model", "mean_rank"],
    )
)
st.altair_chart(c, use_container_width=True)  # type: ignore

# Plot 3 -------------------------------------------------------------------------------------
st.subheader("Number of Wins by Model")
chart_type = st.selectbox("Chart", ["Pie", "Bar"])
if chart_type == "Pie":
    base = alt.Chart(winner_model.group_by("model").agg(pl.len()).collect()).encode(
        color=alt.Color("model", sort=model_sort),
        theta=alt.Theta("len").stack(True),
        tooltip=["model", "len"],
    )
    pie = base.mark_arc(outerRadius=120)
    text = base.mark_text(radius=150, size=10).encode(text="model:N")
    c = pie + text
    st.altair_chart(c, use_container_width=True)  # type: ignore
else:
    c = (
        alt.Chart(winner_model.group_by("model").agg(pl.len()).collect())
        .mark_bar()
        .encode(
            x=alt.X("model", sort=model_sort),
            y="len",
            color=alt.Color("model", sort=model_sort),
            tooltip=["model", "len"],
        )
    )
    st.altair_chart(c, use_container_width=True)

# Plot 4 -------------------------------------------------------------------------------------
st.subheader("Score Composition")
st.info("Bias = (Prediction - Actual)", icon="💡")

d = total_errors.unpivot(
    index=["model", "cutoff_date"],
    value_name="value",
    variable_name="metric",
).collect()

bar = (
    alt.Chart(d.filter(pl.col("metric").is_in(["mae", "bias"])))
    .mark_bar()
    .encode(x="model", y="mean(value)", color="metric", xOffset="metric")
)
line = (
    alt.Chart(d.filter(pl.col("metric") == "score"))
    .mark_line(point=alt.OverlayMarkDef(fill="black"), color="black")
    .encode(x=alt.X("model", sort=model_sort), y="mean(value)")
)
c = alt.layer(bar, line)
st.altair_chart(c, use_container_width=True)  # type: ignore

# Plot 5 -------------------------------------------------------------------------------------
with st.expander("CV"):
    d = (
        total_errors.unpivot(
            index=["model", "cutoff_date"],
            value_name="value",
            variable_name="metric",
        )
        .filter(pl.col("metric").is_in(["bias", "mae"]))
        .collect()
    )
    c = (
        alt.Chart(d)
        .mark_bar()
        .encode(
            x=alt.X("model", sort=model_sort),
            y="value",
            color=alt.Color("model", sort=model_sort),
            opacity="cutoff_date:N",
            xOffset="cutoff_date:N",
        )
        .properties(width=600)
        .facet(row=alt.Facet("metric"))  # type: ignore
    )
    st.altair_chart(c, use_container_width=True)  # type: ignore


# Plot 6 -------------------------------------------------------------------------------------
st.subheader("Score by Horizon")
c = (
    alt.Chart(horizon_errors.collect())
    .mark_bar()
    .encode(
        x="horizon:O",
        y="score",
        color=alt.Color("model", sort=model_sort),
        xOffset=alt.XOffset("model", sort=model_sort),
    )
)
st.altair_chart(c, use_container_width=True)  # type: ignore


# Plot 7 -------------------------------------------------------------------------------------
st.subheader("Competition Score by Sales Scale")
with st.expander("Plot"):
    score_expr = competition_score(cs.starts_with(PRED_PREFIX), pl.col("sales"))
    scores_by_scale = (
        cv_forecast.join(sales, on=["id", "date"])
        .join(scale_category, on="id")
        .group_by("bin")
        .agg(score_expr)
        .sort("bin")
        .unpivot(index="bin", variable_name="model", value_name="score")
        .with_columns(pl.col("model").str.strip_prefix(PRED_PREFIX))
        .collect()
    )

    c = (
        alt.Chart(scores_by_scale)
        .mark_bar()
        .encode(x="model", y="score", color="model")
        .facet(facet="bin", columns=2)
        .resolve_scale(y="independent")
    )
    st.altair_chart(c, use_container_width=True)  # type: ignore
