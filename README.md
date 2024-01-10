# Evolutionary-MOO-for-UAV-routing
This repository delves into the world of UAV routing considering multiple objectives by proposing a fuzzy extension to regular a priori preference settings that converges towards the initially defined preference setting. 
the problem characetristics specificly complicating this problem is that the decision maker cannot be implemented in-operation. This means the preferences need to be implemented prior to the operational execution - that is, preferences have to be implemented a priori and how do we do this in the best possible way?

We do this by investigating the trade-offs one can obtain from implementing preferences a priori through a regular elitist GA and from a posterior through NSGA-II. 
One would have thought that for large scale NP-hard MOO the knowledge of preferences prior to the solution search would ensure that the a priori at least converges faster and towards a better solution relatively to the a posterior NSGA-II approach that identifies a large set of solutions with a large spread around the entire objective space. 
But, we find that despite the need for exploring the entire pareto front, the NSGA-II benefits from its inherent diversity in the populations.  
Consequently, allowing the fuzzy a priori preferences to initially yield the search of a fuzzy pareto front will allow for a faster convergence on average than the two approaches. 

Here is a problem scenario:
![sol0](https://github.com/AlexVasegaard/Evolutionary-MOO-for-UAV-routing/assets/22115067/ef9ead70-de7c-4dab-83af-f4f3126f64ea)

and here is two pareto fronts where the a priori and a posterior are compared:

![GA_stuck1](https://github.com/AlexVasegaard/Evolutionary-MOO-for-UAV-routing/assets/22115067/d0f00c4d-c75c-4ffe-9d4f-df2f7a9b32ed)
![GA_great1](https://github.com/AlexVasegaard/Evolutionary-MOO-for-UAV-routing/assets/22115067/16d820b8-60b9-4334-9792-7872419d5807)
