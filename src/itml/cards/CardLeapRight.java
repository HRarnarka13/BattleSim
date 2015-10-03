package itml.cards;

/**
 *
 *  This class implements a movement Card where the agents moves two squares to the right.
 *
 * @author      Yngvi Bjornsson
 *
 * @version     %I%, %G%
 *
 */
public class CardLeapRight extends Card {
    
    public CardLeapRight() {
        super( "cLeapRight", Card.CardActionType.ctMove, +2, 0,-3, 0, 0, null );
    }

}
