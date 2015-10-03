package itml.cards;

/**
 *
 *  This class implements a movement Card where the agents moves one square to the right.
 *
 * @author      Yngvi Bjornsson
 *
 * @version     %I%, %G%
 *
 */
public class CardMoveRight extends Card {

    public CardMoveRight() {
        super( "cMoveRight", Card.CardActionType.ctMove, +1, 0, -1, 0, 0, null );
    }
}
