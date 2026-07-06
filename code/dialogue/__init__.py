"""Mnemonic-CBR Rebuild — dialogue package (v2).

Multi-turn teacher–student dialogue framework that replaces the v1
static-score evaluation. See ../../REBUILD_SPECIFICATION_v3.md for the
design specification and ../../REBUILD_IMPLEMENTATION_PLAN.md for the
implementation plan.

Public surface:
  - DialogueState, Turn               (state.py)
  - CaseExt                           (state.py)
  - initial_student_utterance         (student.py)
  - StudentSimulator                  (student.py)
  - TeacherGenerator, ARCHITECTURES   (teacher.py)
  - PERSONAS, PERSONA_PRINCIPLES,
    PERSONA_SHORT_DESCRIPTIONS,
    CANONICAL_BASELINE_SENTENCES      (personas.py)
  - run_dialogue                      (loop.py)
  - LLMProvider, StubProvider         (llm_provider.py)
  - clean_mnemonic_engine             (retrieval.py)
"""
