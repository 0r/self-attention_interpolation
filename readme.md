The demo notebook contains and example of the self-attention interpolation architecture used in the below citation:
https://www.frontiersin.org/journals/digital-health/articles/10.3389/fdgth.2025.1657749

```
@ARTICLE{10.3389/fdgth.2025.1657749,
  AUTHOR={Marchal, Noah  and Janes, William E.  and Marushak, Sheila  and Popescu, Mihail  and Song, Xing },
  TITLE={Enhancing ALS progression tracking with semi-supervised ALSFRS-R scores estimated from ambient home health monitoring},
  JOURNAL={Frontiers in Digital Health},
  VOLUME={Volume 7 - 2025},
  YEAR={2025},
  URL={https://www.frontiersin.org/journals/digital-health/articles/10.3389/fdgth.2025.1657749},
  DOI={10.3389/fdgth.2025.1657749},
  ISSN={2673-253X},
  ABSTRACT={IntroductionClinical monitoring of functional decline in amyotrophic lateral sclerosis (ALS) relies on periodic assessments, which may miss critical changes that occur between visits when timely interventions are most beneficial.MethodsTo address this gap, semi-supervised regression models with pseudo-labeling were developed; these models estimated rates of decline by targeting Revised Amyotrophic Lateral Sclerosis Functional Rating Scale (ALSFRS-R) trajectories with continuous in-home sensor data from a three-patient ALS case series. Three model paradigms were compared (individual batch learning and cohort-level batch vs. incremental fine-tuned transfer learning) across linear slope, cubic polynomial, and ensembled self-attention pseudo-label interpolations.ResultsResults showed cohort-level homogeneity across functional domains. For ALSFRS-R subscales, transfer learning reduced the prediction error in 28 of 34 contrasts [mean root mean square error (RMSE) = 0.20 (0.14–0.25)]. However, for composite ALSFRS-R scores, individual batch learning was optimal for two of three participants [mean RMSE = 3.15 (2.24–4.05)]. Self-attention interpolation best captured non-linear progression, providing the lowest subscale-level error [mean RMSE = 0.19 (0.15–0.23)], and outperformed linear and cubic interpolations in 21 of 34 contrasts. Conversely, linear interpolation produced more accurate composite predictions [mean RMSE = 3.13 (2.30–3.95)]. Distinct homogeneity-heterogeneity profiles were identified across domains, with respiratory and speech functions showing patient-specific progression patterns that improved with personalized incremental fine-tuning, while swallowing and dressing functions followed cohort-level trends suited for batch transfer modeling.DiscussionThese findings indicate that dynamically matching learning and pseudo-labeling techniques to functional domain-specific homogeneity-heterogeneity profiles enhances predictive accuracy in tracking ALS progression. As an exploratory pilot, these results reflect case-level observations rather than population-wide effects. Integrating adaptive model selection into sensor platforms may enable timely interventions as a method for scalable deployment in future multi-center studies.}}
```

Marchal N, Janes WE, Marushak S, Popescu M and Song X (2025) Enhancing ALS progression tracking with semi-supervised ALSFRS-R scores estimated from ambient home health monitoring. Front. Digit. Health 7:1657749. doi: 10.3389/fdgth.2025.1657749
