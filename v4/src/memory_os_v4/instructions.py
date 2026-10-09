"""MCP server instructions injected into every session (``Server(instructions=...)``)."""

MEMORY_INSTRUCTIONS = """Memory OS v4 — persistent memory. Follow these rules.

1. SESSION START: call memory_session_brief before answering (focus_query for the topic).
   Returns core blocks, recent episodes, key facts, procedures, pending items.
2. RECALL: before guessing, call memory_search (at_time="YYYY-MM-DD" for "what was true on X")
   or memory_timeline ("how did X evolve"). For a comparison/counting/multi-part question also
   pass queries: 2-4 keyword-rich sub-questions naming the person/entity each needs; and
   hypotheses: 1-2 sentences that would directly answer it. Both fuse with the query via RRF.
3. SAVE after significant work, every ~10 exchanges, and before ending: memory_chat_save with
   messages, a memo summary, and facts (ADD-only, atomic):
   - One sentence, explicit subject, no pronouns. Never delete — state the new fact with its
     date; the server supersedes the old one deterministically.
   - Specific entities ("Nisha's dad" not "dad"); a changed detail ("senior" vs "software
     engineer") is an update, add both.
   - Ground relative dates to the message's own date ("yesterday" -> absolute event_time).
   - Include: preferences/constraints, decisions+rationale, results with numbers, file/project
     locations, tasks (tags=["todo"]), lessons (kind="procedure"), agent actions as facts.
   - Add keywords to each message and fact: 3-8 lowercase topical words (nouns, activities,
     places, names) — improves recall, never shown to the user.
   - Caveat uncertain facts with a lower confidence instead of omitting them.
4. BLOCKS: memory_block_set (mode="replace") when profile/project/rules change; terse, not
   accumulating.
5. CONSOLIDATE: memory_consolidate when the brief reports >2000 items, or weekly; write
   summary_proposals via memory_add_facts(kind="summary").

Example memory_chat_save:
{"messages": [{"role": "user", "content": "Ship the v4 MCP server today.",
               "keywords": ["mcp", "server", "shipping"]}],
 "summary": "Built the v4 MCP server: 16 tools, tests green.",
 "facts": [{"text": "Romain shipped the Memory OS v4 MCP server on 2026-09-12.",
            "entities": ["Romain", "Memory OS v4"], "event_time": "2026-09-12",
            "keywords": ["mcp", "server", "release"]}]}

Example memory_search with planning:
{"query": "Did Romain and Nisha like the same restaurant?",
 "queries": ["Romain favorite restaurant", "Nisha favorite restaurant"],
 "hypotheses": ["Romain and Nisha both liked the same restaurant."]}
"""

# Facet channel addendum (2026-09-19): appended to `MEMORY_INSTRUCTIONS` by
# `build_instructions()` ONLY when `config.facet_channel` is on -- see
# config.py's `facet_channel`/`facet_weight` for why it ships off by
# default. `MEMORY_INSTRUCTIONS` itself must stay byte-identical with the
# flag off (there's a test on its length/content), so this lives in its own
# string instead of being folded into it. Kept <= 900 chars (task brief).
FACET_INSTRUCTIONS_ADDENDUM = (
    "\n6. FACETS (channel on): save -> facets: [{subject,predicate,object}] per message/fact; "
    "predicate in: achievement_award, age_birthday, belief_value_opinion, book_read, "
    "challenge_problem, child, colleague_mentor, creative_work, dislike_fear, education_course, "
    "employer_business, event_attended, family_member, feeling_emotion, food_drink_recipe, "
    "friend, game_played, gender_identity, gift_given_received, goal_plan, habit_routine, "
    "health_condition, hobby_interest, job_career, life_event_milestone, movie_show_music, "
    "name_alias, nationality_language, origin_hometown, outdoor_activity, partner, person_met, "
    "pet, possession_purchase, preference_like, project_work, residence, skill_ability, "
    "sport_exercise, support_advice, trip_location_visited, vehicle, volunteering_cause; else "
    '"other" (unindexed). search -> facets: [{subject,predicate}] (no object); fused into '
    "ranking.\n"
)


def build_instructions(config) -> str:
    """`MEMORY_INSTRUCTIONS`, plus `FACET_INSTRUCTIONS_ADDENDUM` when
    `config.facet_channel` is on. With the flag off (the default) this
    returns `MEMORY_INSTRUCTIONS` unchanged -- byte-identical, per the
    existing length/content test."""
    if getattr(config, "facet_channel", False):
        return MEMORY_INSTRUCTIONS + FACET_INSTRUCTIONS_ADDENDUM
    return MEMORY_INSTRUCTIONS
