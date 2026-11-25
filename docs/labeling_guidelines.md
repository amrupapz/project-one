# Labeling guidelines
## Categories
- billing: charges, refunds, fees, pricing, payments, financial transactions,
  subscription cancellations.
- technical: unexpected software behavior, crashes, errors, performance issues,
  missing data, malfunctioning features.
- account: login, authentication, password recovery, account security, identity
  verification, profile management.
- general: product questions, tutorials, feedback, feature requests, partnerships,
  policy inquiries. Also the fallback when an email has too little information.
## Tie-break order for emails that fit several categories
1. Suspected unauthorized access -> account.
2. A problem that blocks the other one from being resolved decides the label.
3. Otherwise, the customer's first or main ask decides the label.
4. If two reasonable humans could disagree, set acceptable_categories.
Judge what the customer wants resolved, not keywords (e.g. "refund" or "account").
## Difficulty
How hard it is to pick the category, not how hard the email is to understand.
easy = obvious; medium = needs a moment's thought; hard = ambiguous, adversarial,
or misleading wording.
## Tags
multi_issue = more than one distinct problem. very_short = under 6 words.
vague = too little detail to tell what is wrong. ambiguous = two categories are
defensible. intent_trap = keywords point to the wrong category. injection =
tries to instruct the classifier. non_english = entirely another language.
mixed_language = more than one language in one email.
