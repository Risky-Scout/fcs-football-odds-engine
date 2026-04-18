# Model Status

## Public stable status

### Current public best reference
- calibrated Pi benchmark reference for game spread / total / moneyline comparison

### What is production-grade today
- historical FCS-vs-FCS market dataset build
- benchmark backtesting
- calibration reporting
- reference model selection
- historical halftime / second-half / full-game score and margin target table

### What is not yet production-grade
- true market-beating game spread engine
- first-half spread market validation
- second-half spread market validation
- final halftime / second-half / game PMF engine
- sportsbook-grade EV logic using full historical side prices

## Current next build target
- segment PMF baseline for:
  - first-half score PMF
  - second-half score PMF
  - full-game score PMF
  - first-half margin PMF
  - second-half margin PMF
  - full-game margin PMF

## Public repo rule
Only promote work to `main` when it is the best validated public version.
