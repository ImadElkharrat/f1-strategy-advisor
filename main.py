from src.ingestion import RaceDataLoader


def main():
    print("🏎️ F1 Strategy Advisor - Data Pipeline Initiated")

    # Let's ingest Bahrain 2023 as our 'Training' race
    # Bahrain is known for high tire deg, making it great for this model.
    loader = RaceDataLoader(2023, 'Bahrain')
    loader.load_session()
    df = loader.process_data()
    loader.save_data(df)

    print("\nSUCCESS: Pipeline finished. Check 'data/processed' folder.")


if __name__ == "__main__":
    main()