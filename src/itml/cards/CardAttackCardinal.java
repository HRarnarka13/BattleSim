package itml.cards;

import itml.simulator.Coordinate;

/**
 *
 *  This class implements a Card that attacks the square the agent is on as well as the
 *  four adjacent cardinal directions.
 *
 * @author      Yngvi Bjornsson
 *
 * @version     %I%, %G%
 *
 */
public class CardAttackCardinal extends Card {

    private static Coordinate[] cardinalAttacks = { coO, coN, coE, coS, coW };

    public CardAttackCardinal() {
        super( "cAttackCardinal", Card.CardActionType.ctAttack, 0, 0, -2, 1, 0, cardinalAttacks );
    }
}
