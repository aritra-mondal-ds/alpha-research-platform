# Trading Bot Monorepo

A unified repository for multiple trading bot applications.

## 📁 Directory Structure

```
Trading-Bot/
├── crypto/          - Cryptocurrency Trading Bot
├── forex/           - Forex Trading Bot (Coming Soon)
├── commodities/     - Commodities Trading Bot (Coming Soon)
└── README.md
```

## 🤖 Applications

### Crypto Trading Bot (`/crypto`)
Automated trading bot for cryptocurrency markets.

**Current Status:** Active Development
**Files:**
- `CryptoTraderBot.ipynb` - Main trading bot notebook
- `environment.ipynb` - Environment setup notebook
- `requirements.txt` - Python dependencies

### Forex Trading Bot (`/forex`)
Automated trading bot for foreign exchange markets.

**Current Status:** Placeholder

### Commodities Trading Bot (`/commodities`)
Automated trading bot for commodity markets.

**Current Status:** Placeholder

## 🚀 Getting Started

1. Navigate to the app directory you want to work with:
   ```bash
   cd crypto  # or forex, or commodities
   ```

2. Set up the environment:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the trading bot:
   ```bash
   jupyter notebook CryptoTraderBot.ipynb
   ```

## 📝 Notes

- Each app directory is self-contained with its own dependencies
- All apps share this unified Git repository
- Future apps (Forex, Commodities) will follow the same structure

## 📞 Contributing

- Create feature branches for new developments
- Commit changes to the respective app directory
- Ensure each app's `requirements.txt` is updated with new dependencies
