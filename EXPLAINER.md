# Explainer: from PD to capital on Lending Club loans
Author: Victor Yu

## 1. The question
Given a portfolio of consumer loans, how likely is each borrower to default, how much will the portfolio lose on average, how bad can losses get in a downturn, how much capital should be held, what should be reserved under IFRS 9, and how far can the model be trusted?

## 2. The data
- [All Lending Club loan data](https://www.kaggle.com/datasets/wordsforthewise/lending-club) (Kaggle, CC0).
- Sample: 36-month loans issued 2012 to 2014 that have finished (fully paid or charged off), so every outcome is known. 306,462 loans; 250 removed by simple filters (a key field blank, dti outside 0 to 60, rare home-ownership types). **306,212 loans**, default rate 13.2%.
- Split: sorted by loan id, random 60/20/20 with seed 2024: **dev 183,909 / test 61,345 / holdout 60,958**. Holdout funded amount: $765.3M.
- Default = charged off or in default.

## 3. Method in plain words
1. **PD**: logistic regression of default on application data: FICO, dti, income, loan amount, job tenure (with a flag when blank), revolving utilisation, recent inquiries, past delinquencies, credit-history length, home ownership, income verification and loan purpose. Lending Club's grade and interest rate are left out, because they are Lending Club's own risk assessment.
2. **EAD**: for defaulted loans, EAD = funded amount − principal repaid. Predicted EAD for any loan = funded amount × average EAD % of its grade (dev).
3. **LGD**: LGD = 1 − (recoveries − collection fee) / EAD, clipped to [0, 1]. Two models on defaulted dev loans (grade means; linear regression on borrower traits), compared on defaulted test loans.
4. **Default timing**: a loan defaults when it is 90 days past due (3 months after its last payment). F(t) = share of lifetime defaults that happen by month t. Cumulative PD by month t = lifetime PD × F(t).
5. **Expected Loss** per holdout loan = PD × LGD × EAD, compared with the actual principal loss, in total and by PD decile.
6. **Credit VaR and ES**: one-factor Gaussian (Vasicek) model. Given the economy factor z, each loan's PD becomes PD(z) = Φ((Φ⁻¹(PD) − √ρ·z) / √(1 − ρ)), and portfolio loss is L(z) = Σ PD_i(z)·LGD_i·EAD_i. L(z) is computed on a fine grid and read off for 1,000,000 simulated values of z. With 61,000 small loans, idiosyncratic noise averages out, so no per-loan simulation is needed. VaR = loss quantile; ES = average loss beyond the VaR.
7. **Basel IRB "other retail"**: R = 0.03·w + 0.16·(1 − w), w = (1 − e^(−35·PD)) / (1 − e^(−35)); K = LGD·[Φ((Φ⁻¹(PD) + √R·Φ⁻¹(0.999)) / √(1 − R)) − PD]; RWA = 12.5·K·EAD. 1-year PD = lifetime PD × F(12); 5 bp floor.
8. **Stress tests**: PD ×1.5, PD ×2, LGD +10 points, and PD ×2 with LGD +10 points.
9. **Validation**: AUC, Gini = 2·AUC − 1, KS, decile calibration with z-scores, and the population stability index (PSI).
10. **Vintage and concentration**: default and loss rates by issue quarter; exposure HHI by state, purpose and grade.
11. **IFRS 9**: staging and expected credit loss at a reporting date 12 months after issue.

## 4. Choices
| Choice | What I picked | Why |
| --- | --- | --- |
| PD inputs | Application data only | A bank scoring new applicants would not have Lending Club's grade or rate |
| Blank job tenure | Set to 0 and flagged | Keeps 16,000 loans; the flag itself raises the odds of default by 39% |
| EAD predictor | Grade average EAD % | EAD % rises steadily by grade (51% to 70%) but is flat across FICO bands |
| LGD model | Grade means | Rule set in advance: the regression must cut test RMSE by at least 0.5%. It cut it by 0.35% (R² 0.009) |
| Default date | 90 days past due | The Basel and IFRS 9 definition; makes Stage 3 and the timing curve consistent |
| Correlation ρ | 8% main run; 3% and 15% as sensitivity | Brackets the Basel retail range (3% to 16%); the Basel formula averages 8.2% on this book |
| VaR horizon | Life of the loan (36 months) | Matches the actual losses I can check against; the IRB comparison uses 1-year PDs |
| Loss definition | Principal lost net of recoveries and fees; no discounting | Observable for every loan and consistent between EL and actual loss |
| Stage 3 / Stage 2 backstops | 3+ missed payments = Stage 3; 2 missed = Stage 2 | IFRS 9's 90- and 30-days-past-due presumptions |
| Significant increase in credit risk (SICR) proxy | PD ≥ 2 × dev default rate (26.4%) | No updated scores in the data. On test it flags 3.6% of loans, which default 28.8% vs 12.8% |

## 5. Results

### PD model
| Measure | Dev | Test | Holdout |
| --- | --- | --- | --- |
| AUC | 0.654 | 0.654 | 0.650 |
| Gini | 0.307 | 0.309 | 0.300 |
| KS | 0.220 | 0.231 | 0.222 |

Odds of default for a typical borrower: +10 FICO points ×0.89; +5 dti points ×1.10; income doubles ×0.70; loan amount doubles ×1.21; one more inquiry in 6 months ×1.17; 5 more years of credit history ×0.96; job tenure left blank ×1.39.
- Calibration (holdout): predicted 13.2% vs actual 13.3% overall. Three deciles are off by more than 2 standard errors: decile 9 under-predicted (19.7% vs 21.1%), decile 10 over-predicted (26.2% vs 24.9%), decile 5 over-predicted (11.6% vs 10.8%).
- PSI dev vs holdout 0.0001 (expected from a random split); 2012 vs 2014 loans 0.015, stable.

### EAD and LGD
| Result | Number | Meaning |
| --- | --- | --- |
| EAD at default | 57.4% of funded; grade A 51.5% → grade G 69.7% | Riskier borrowers default earlier, before repaying much. Using the full funded amount would overstate EAD by about 74% |
| LGD | Mean 88.8%, median 88.5%; 16.2% of defaults recover nothing | Unsecured loans: charged-off debt recovers about 11 cents on the dollar |
| LGD by grade | 87.4% to 89.3% | Almost flat: grade drives PD and EAD, not LGD |
| LGD model on test | RMSE 0.1135 (grade means) vs 0.1131 (regression) | Borrower traits barely predict LGD |

### Default timing and vintages
- 22.7% of lifetime defaults happen by month 12 and 66.6% by month 24.
- Vintages (dev): default rate 12.8% to 14.3% for 2012 issues, a low of 12.1% in 2013 Q3, and 14.5% for 2014 Q4; loss rate 7.8% → 5.8% → 7.2%. The predicted PD rose from 11.5% to 13.8%, so the model tracks the later deterioration only partly.

### Expected Loss (holdout)
| Result | Number |
| --- | --- |
| Total | EL **$49.16M (6.42% of funded)** vs actual **$49.63M (6.48%)**; ratio 1.009 |
| Components | Default rate 13.2% vs 13.3%; EAD on defaulted loans $55.6M vs $55.7M; LGD 88.8% vs 89.1% |
| By PD decile | Actual / predicted between 0.90 and 1.12 |

### Credit VaR, ES and capital (holdout, $M, loan-life horizon)
| ρ | EL | 99% VaR | 99% ES | 99.9% VaR | 99.9% ES |
| --- | --- | --- | --- | --- | --- |
| 3% | 49.2 | 85.7 | 92.5 | 101.3 | 107.2 |
| 8% | 49.2 | 114.9 | 128.1 | 145.2 | 156.6 |
| 15% | 49.2 | 147.2 | 167.5 | 193.5 | 209.9 |

- Economic capital at ρ = 8% (99.9% VaR − EL): **$96.0M, 12.5% of funded**. EL does not depend on ρ; the tail does. Going from 3% to 15% nearly doubles the 99.9% VaR.
- **Basel IRB check**: average 1-year PD 3.0%, EAD-weighted R 8.2%, capital **$40.4M**, RWA $505M (119% of EAD). My simulation with the same inputs gives a 99.9% VaR − EL of **$40.8M**, within 1%. The IRB formula is the closed-form version of the same one-factor model, so this confirms the simulation.

### Stress tests (holdout, ρ = 8%, $M)
| Scenario | EL | EL % of funded | 99.9% VaR | 99.9% ES |
| --- | --- | --- | --- | --- |
| Base | 49.2 | 6.4% | 145.2 | 156.6 |
| PD ×1.5 | 73.7 | 9.6% | 184.5 | 196.0 |
| PD ×2 | 98.3 | 12.8% | 216.1 | 227.1 |
| LGD +10 pts | 54.7 | 7.1% | 161.5 | 174.2 |
| PD ×2, LGD +10 pts | 109.4 | 14.3% | 240.4 | 252.6 |

EL scales one-for-one with PD, but VaR rises less (PD ×2 lifts VaR by 49%), because part of the stress is already in the tail. LGD is already near 89%, so +10 points adds little.

### Concentration (holdout exposure)
States HHI 0.059 (equivalent to 17 equal states; CA 16.2%, NY 8.7%, TX 8.7%). Purpose HHI 0.435 (debt consolidation 60.2%, credit card 26.2%). Grade HHI 0.260 (B 36.1%). Geography is well spread; the book is concentrated in debt-refinancing products.

### IFRS 9, 12 months after issue (holdout)
| Stage | Loans | Balance | ECL | Coverage | Later default rate |
| --- | --- | --- | --- | --- | --- |
| Stage 1 | 49,381 | $447.1M | $17.6M | 3.9% | 11.2% |
| Stage 2 | 2,068 | $15.1M | $1.8M | 12.2% | 33.4% |
| Stage 3 | 1,882 | $19.0M | $16.8M | 88.7% | 100% |
| Total | 53,331 | $481.1M | $36.3M | 7.5% | 15.2% |

7,627 loans were repaid by month 12. For performing loans, ECL is $19.1M if all are Stage 1, $24.6M if all are lifetime, and $19.5M with staging.

## 6. Limits
- **Stage 2 is under-reserved**: coverage 12.2%, but 33.4% of these loans later default (about 30% loss at 89% LGD). The PDs come from origination and are not updated for missed payments. A production IFRS 9 model would use behavioural PDs or roll rates, probability-weighted macro scenarios, and discounting at the effective interest rate.
- **Moderate discrimination**: AUC 0.65 from application data. Lending Club's own grade ranks better, so the model would be a challenger, not a replacement.
- **Calibration gaps** in deciles 5, 9 and 10 (|z| 2.1 to 2.7) would need a recalibration step before use in pricing.
- **Finished, funded loans only**: rejected applicants and loans still running in 2018 are not seen.
- **LGD** ignores the time value of recoveries and costs other than the collection fee; recoveries reflect Lending Club's debt sales.
- **EAD for capital**: I use the expected balance at default; Basel would use the current balance of a term loan.
- **Default timing**: the few loans that reach 90 days past due after month 36 are counted in month 36, which causes the small step at the end of the term-structure chart.
- **ρ is assumed, not estimated**: 2012 to 2014 was a benign period with no downturn to estimate it from.

## 7. Interview questions and answers
1. **Why does EL match actual loss so closely if the AUC is only 0.65?**
   AUC measures ranking; EL needs calibration. A model can rank borrowers only moderately well and still have the right average level. EL depends on the average PD, LGD and EAD being right, and each matched within about 1%. For approving individual loans, the modest AUC matters much more.

2. **What is the difference between VaR and Expected Shortfall, and why do regulators prefer ES?**
   VaR is the loss you exceed only 0.1% of the time; ES is the average loss in that worst 0.1%. VaR says nothing about how bad the tail is beyond the cut-off and is not always sub-additive. ES captures tail severity and is coherent, which is why Basel's market-risk rules (FRTB) moved to ES at 97.5%. Here the 99.9% ES is $11M above the VaR at ρ = 8%.

3. **What does asset correlation ρ represent, and why does it matter so much?**
   It is the share of each borrower's credit quality driven by the common economy rather than personal circumstances. EL does not depend on ρ at all, but the tail does: moving ρ from 3% to 15% raises the 99.9% VaR from $101M to $193M. Defaults cluster when the economy turns, so ρ is the key assumption in any capital number.

4. **How did you check that your simulation is right?**
   Two checks. The mean simulated loss equals the analytical EL ($49.2M). And with 1-year PDs and the Basel correlations, my simulated 99.9% VaR minus EL ($40.8M) matches the Basel IRB closed-form capital ($40.4M) within 1%. The IRB formula is the same one-factor model solved analytically, so the two should agree.

5. **Why did you keep the simpler LGD model when the regression was slightly better?**
   I set the rule before looking at test results: the regression had to cut test RMSE by at least 0.5%. It cut it by 0.35%, with R² of 0.009, so borrower traits explain almost nothing about LGD. The simpler model is easier to validate and explain, and fixing the rule in advance avoids picking a model after seeing the answer.

6. **Your PSI is almost zero. Does that prove the model is stable?**
   No. Dev and holdout come from a random split of the same years, so a PSI near zero is expected. The more meaningful check is over time: 2012 vs 2014 loans give 0.015, still stable. But stable inputs do not guarantee stable performance: vintage default rates moved between 12.1% and 14.5% while predicted PDs moved less. In production I would track PSI together with realised default rates by vintage.

7. **How would you make this a production IFRS 9 model, and what is missing?**
   I have a lifetime PD term structure, 12-month and lifetime ECL, and staging with the 30- and 90-days-past-due backstops. What is missing: PDs updated after origination (my Stage 2 loans default 33.4% of the time but are reserved at 12.2%), probability-weighted macro scenarios, and discounting at the effective interest rate.

8. **What would you tell a credit risk committee about this book?**
   Expected losses are about 6.4% of funded and the model predicts them accurately. At an 8% correlation, the 99.9% loss could reach about $145M, or 19% of funded. Losses are most sensitive to PD: a doubling of default rates, as in a recession, doubles EL to $98M. Geography is well spread, but 86% of exposure is debt or credit-card refinancing, so the book depends on one product type and on consumer credit conditions.
