"""Concreteness detection must be CONTENT-AGNOSTIC (no hardcoded domain allow-lists, per the
project rule). A sentence is concrete because it carries a verifiable anchor that holds in any
subject -- never because it matches a domain's vocabulary.
"""

from __future__ import annotations

from poc.detect.layer3_scoring import (
    _sentence_has_concrete_or_context as concrete,
    _has_author_owned_context as owned,
    estimate_lived_detail_risk,
)


def test_credits_real_anchors_in_any_domain():
    assert concrete("The reaction yielded 4.2 grams of product.")            # number (chemistry)
    assert concrete("After the valve opens, pressure drops sharply.")        # situational (engineering)
    assert concrete("For example, a tenant can withhold rent here.")         # example (law)
    assert concrete('The witness said "I never signed it."')                 # quote
    assert concrete("Smith (2019) reported the same trend.")                 # citation
    assert concrete("When a buyer defaults, the lender forecloses.")         # temporal clause (finance)


def test_does_not_credit_bare_generic_claims():
    assert not concrete("Technology is changing the world rapidly.")
    assert not concrete("Good leadership matters in every organization.")
    assert not concrete("The system should be efficient and reliable.")


def test_no_domain_hardcoding():
    # The removed allow-list hardcoded hairdressing/education vocab. Merely naming a domain must NOT
    # auto-credit a sentence -- only a real anchor does.
    assert not concrete("Hairdressing requires skill and practice.")
    assert not concrete("Education is important for society.")
    assert not concrete("Students learn many things at school.")


def test_lived_detail_author_context_is_agnostic():
    # First-hand experience in ANY domain is credited.
    assert owned("In my clinic I observed the rhythm change after the dose.")
    assert owned("For example, the reaction failed on the third attempt.")
    # Domain jargon alone no longer auto-credits (de-hardcoded).
    assert not owned("The haircut used a comb and scissor on the mannequin.")
    assert not owned("The policy improves educational outcomes.")


def test_lived_detail_risk_rewards_first_person_specifics():
    lived = ("In my clinic I noticed three patients react differently. "
             "When the dose changed, I observed the rhythm settle. "
             "For example, after an hour the tremor eased.")
    generic = ("Healthcare is important for society. "
               "Good treatment matters in every setting. "
               "The system should be efficient and reliable.")
    assert estimate_lived_detail_risk(lived) < estimate_lived_detail_risk(generic)
