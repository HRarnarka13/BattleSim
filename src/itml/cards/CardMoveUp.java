package itml.cards;

/**
 *
 *  This class implements a movement Card where the agents moves one square up.
 *
 * @author      Yngvi Bjornsson
 *
 * @version     %I%, %G%
 *
 */
public class CardMoveUp extends Card {

    public CardMoveUp() {
        super( "cMoveUp", Card.CardActionType.ctMove, 0, +1, -1, 0, 0, null );
    }
}
