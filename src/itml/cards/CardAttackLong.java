package itml.cards;

import itml.simulator.Coordinate;

/**
 *
 *  This class implements a Card that attacks the square the agent is on as well as the next
 *  two squares (back and forth) horizontally.
 *
 * @author      Yngvi Bjornsson
 *
 * @version     %I%, %G%
 *
 */

public class CardAttackLong extends Card {

    private static Coordinate[] cardinalLongline = { coO, coE, coW, coE2, coW2 };

    public CardAttackLong() {
        super( "cAttackLong", Card.CardActionType.ctAttack, 0, 0,-2, 1, 0, cardinalLongline );
    }
}
