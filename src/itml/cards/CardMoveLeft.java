package itml.cards;

/**
 *
 *  This class implements a movement Card where the agents moves one square to the left.
 *
 * @author      Yngvi Bjornsson
 *
 * @version     %I%, %G%
 *
 */

public class CardMoveLeft extends Card {

    public CardMoveLeft() {
        super( "cMoveLeft", Card.CardActionType.ctMove, -1, 0, -1, 0, 0, null );
    }
}
