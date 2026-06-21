import MetaTrader5 as mt5
from datetime import datetime, timezone


def main():
    if not mt5.initialize():
        print(f"MT5 initialize() failed: {mt5.last_error()}")
        return

    try:
        positions = mt5.positions_get()

        print(
            f'{"Symbol":<10} {"Ticket":<12} {"Type":<5} {"Volume":<8} '
            f'{"Open Time (UTC)":<20} {"SL":<10} {"TP":<10} {"Profit":<10} {"Magic"}'
        )
        print('-' * 100)

        if positions:
            for p in positions:
                open_time = datetime.fromtimestamp(p.time, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')
                position_type = 'BUY' if p.type == 0 else 'SELL'

                print(
                    f'{p.symbol:<10} {p.ticket:<12} {position_type:<5} {p.volume:<8} '
                    f'{open_time:<20} {p.sl:<10} {p.tp:<10} {p.profit:<+10.2f} {p.magic}'
                )
        else:
            print('No open positions.')

    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()