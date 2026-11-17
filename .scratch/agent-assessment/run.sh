#!/usr/bin/env bash
# Two independent read-only assessments from BRIEF.md, in parallel: agy (Gemini 4 Argon High) and codex (gpt-6.1-terra, xhigh).
cd /mnt/School/PhD/AI231/AI-231-ME4 || exit 1
O=.scratch/agent-assessment
R=/tmp/claude-1000/-mnt-School-PhD-AI231-AI-231-ME4/b00e114e-0f9a-49aa-a4b8-37acc3feb7e2/scratchpad/res
( timeout 5400 agy --mode plan --model gemini-4-argon-high --print-timeout 85m \
    --add-dir /mnt/Obsidian/Claude/second-brain/resources --add-dir /mnt/Others/learning/extra-resources --add-dir $R \
    -p "$(cat $O/BRIEF.md)" > $O/AGY-ARGON.md 2> $O/agy.err; echo "exit $?" > $O/agy.done ) &
( timeout 5400 codex exec -m gpt-6.1-terra -c model_reasoning_effort=xhigh -s read-only --ephemeral \
    --add-dir /mnt/Obsidian/Claude/second-brain/resources --add-dir /mnt/Others/learning/extra-resources --add-dir $R \
    -o $O/CODEX-TERRA.md "$(cat $O/BRIEF.md)" > $O/codex.log 2> $O/codex.err; echo "exit $?" > $O/codex.done ) &
wait
echo all done
