package itml.cards;

import itml.simulator.Coordinate;

/**
 *
 *  This class implements a Card that attacks the square the agent is on as well as the
 *  four adjacent diagonal directions.
 *
 * @author      Yngvi Bjornsson
 *
 * @version     %I%, %G%
 *
 */

public class CardAttackDiagonal extends Card {

    private static Coordinate[] cardinalDiagonal = { coO, coNE, coSE, coSW, coNW };

    public CardAttackDiagonal() {
        super( "cAttackDiagonal", Card.CardActionType.ctAttack, 0, 0,-2, 1, 0, cardinalDiagonal );
    }
}
