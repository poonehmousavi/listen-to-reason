"""Run every self-test that needs no model, no GPU and no data.

    python -m l2r.selftest
"""
from l2r import events
from l2r.dataset import annotator, experts, index, schema, segment
from l2r.router import data as router_data, features, labelled_music, train
from l2r.tree import build as tree, guards, llm_clean


def main():
    s = schema.load_schema()
    segment.selftest(s); index.selftest(); experts.selftest(s); annotator.selftest(s)
    guards.selftest(); tree.selftest(s); llm_clean.selftest()
    router_data.selftest(); features.selftest(); labelled_music.selftest(); train.selftest()
    events.selftest()
    print("all self-tests passed")


if __name__ == "__main__":
    main()
