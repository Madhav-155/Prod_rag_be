"""Explicit purge of expired conversations and cascading turn/event rows only."""

import argparse

from rag_backend.config import Settings
from rag_backend.conversations.repository import ConversationRepository
from rag_backend.storage.database import create_database_engine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--purge-expired', action='store_true', required=True)
    parser.parse_args()
    engine = create_database_engine(Settings())
    try:
        print(f'Purged expired conversations: {ConversationRepository(engine).purge_expired()}')
    finally:
        engine.dispose()


if __name__ == '__main__':
    main()
