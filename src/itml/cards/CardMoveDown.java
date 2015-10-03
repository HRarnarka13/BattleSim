package itml.cards;

/**
 *
 *  This class implements a movement Card where the agents moves one square down.
 *
 * @author      Yngvi Bjornsson
 *
 * @version     %I%, %G%
 *
 */
public class CardMoveDown extends Card {

    public CardMoveDown() {
        super( "cMoveDown", Card.CardActionType.ctMove, 0, -1, -1, 0, 0, null );
    }
}
