"""
AI Reconciliation Service
"""

from typing import List, Tuple, Dict
import asyncio
from datetime import datetime, timedelta


class ReconciliationService:
    """Service for AI-powered transaction reconciliation"""

    @staticmethod
    def calculate_similarity(
        bank_tx: dict,
        ledger_tx: dict,
        amount_threshold: float = 0.01,
    ) -> float:
        """
        Calculate similarity score between two transactions.
        Returns: 0.0 - 1.0
        """
        score = 0.0
        
        # Amount matching (40% weight)
        amount_diff = abs(bank_tx.get("amount", 0) - ledger_tx.get("amount", 0))
        max_amount = max(abs(bank_tx.get("amount", 0)), abs(ledger_tx.get("amount", 0)))
        
        if max_amount > 0:
            amount_similarity = 1.0 - min(amount_diff / max_amount, 1.0)
            score += amount_similarity * 0.4
        
        # Date matching (30% weight)
        bank_date = bank_tx.get("date")
        ledger_date = ledger_tx.get("date")
        
        if bank_date and ledger_date:
            date_diff = abs((bank_date - ledger_date).days)
            date_similarity = 1.0 - min(date_diff / 30.0, 1.0)  # 30 day window
            score += date_similarity * 0.3
        
        # Description matching (20% weight)
        bank_desc = str(bank_tx.get("description", "")).lower()
        ledger_desc = str(ledger_tx.get("description", "")).lower()
        
        if bank_desc and ledger_desc:
            common_words = len(set(bank_desc.split()) & set(ledger_desc.split()))
            total_words = len(set(bank_desc.split()) | set(ledger_desc.split()))
            
            if total_words > 0:
                desc_similarity = common_words / total_words
                score += desc_similarity * 0.2
        
        # Account matching (10% weight)
        if bank_tx.get("account") == ledger_tx.get("account"):
            score += 0.1
        
        return round(score, 3)

    @staticmethod
    async def reconcile_batch(
        bank_transactions: List[dict],
        ledger_transactions: List[dict],
        confidence_threshold: float = 0.6,
    ) -> Dict:
        """
        Reconcile a batch of transactions.
        
        Returns:
            {
                "matches": [...],
                "unmatched_bank": [...],
                "unmatched_ledger": [...],
                "total_matched_amount": float,
            }
        """
        matches = []
        matched_bank_ids = set()
        matched_ledger_ids = set()
        
        # Calculate similarity for all pairs
        for bank_tx in bank_transactions:
            best_match = None
            best_score = confidence_threshold
            best_ledger_idx = None
            
            for ledger_idx, ledger_tx in enumerate(ledger_transactions):
                if ledger_idx in matched_ledger_ids:
                    continue
                
                score = ReconciliationService.calculate_similarity(bank_tx, ledger_tx)
                
                if score > best_score:
                    best_score = score
                    best_match = ledger_tx
                    best_ledger_idx = ledger_idx
            
            if best_match:
                matches.append({
                    "bank_transaction_id": bank_tx.get("id"),
                    "ledger_transaction_id": best_match.get("id"),
                    "confidence": best_score,
                    "amount_difference": abs(
                        bank_tx.get("amount", 0) - best_match.get("amount", 0)
                    ),
                })
                matched_bank_ids.add(bank_tx.get("id"))
                matched_ledger_ids.add(best_ledger_idx)
        
        # Calculate total matched amount
        total_matched = sum(
            match["amount_difference"] == 0
            for match in matches
        )
        
        return {
            "matches": matches,
            "unmatched_bank": len(bank_transactions) - len(matched_bank_ids),
            "unmatched_ledger": len(ledger_transactions) - len(matched_ledger_ids),
            "total_matched_amount": len([m for m in matches if m["amount_difference"] == 0]),
        }
