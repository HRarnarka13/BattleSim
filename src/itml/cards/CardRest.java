package itml.cards;

/**
 *
 *  This class implements a Card where the agents rests, replenishing its stamina.
 *
 * @author      Yngvi Bjornsson
 *
 * @version     %I%, %G%
 *
 */
public class CardRest extends Card {

    public CardRest() {
        super( "cRest", Card.CardActionType.ctMove, 0, 0, 3, 0, 0, null );
    }
}
