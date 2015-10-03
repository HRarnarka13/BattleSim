package itml.cards;

/**
 *
 *  This class implements a Card that (partially) defends off the opponents attack.
 *
 * @author      Yngvi Bjornsson
 *
 * @version     %I%, %G%
 *
 */
public class CardDefend extends Card {

    public CardDefend() {
        super( "cDefend", Card.CardActionType.ctDefend, 0, 0, -2, 0, 2, null );
    }
}
