With binary_pred as (
    Select
        *
    From {{ref('batch_binary_pred')}}
)
Select
    Player_ID as player_id,
    Player as player,
    "Top-League"     as top_league,
    "Non-Top-League" as non_top_league,
    Creation_Timestamp as creation_timestamp
From binary_pred