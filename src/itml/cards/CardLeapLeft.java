package itml.cards;

/**
 *
 *  This class implements a movement Card where the agents moves two squares to the left.
 *
 * @author      Yngvi Bjornsson
 *
 * @version     %I%, %G%
 *
 */
public class CardLeapLeft extends Card {

    public CardLeapLeft() {
        super( "cLeapLeft", Card.CardActionType.ctMove, -2, 0,-3, 0, 0, null );
    }
}
