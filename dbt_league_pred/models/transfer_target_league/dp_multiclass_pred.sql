With multi_pred as (
    Select
        *
    From {{ref('batch_multiclass_pred')}}
)
Select
    Player_ID as player_id,
    Player as player,
    Bundesliga as bundesliga,
    LaLiga as laliga,
    list_extract(["Ligue 1"], 1) as ligue_1,
    list_extract(["Premier League"], 1) as premier_league,
    list_extract(["Serie A"], 1) as serie_a,
    creation_timestamp
From multi_pred