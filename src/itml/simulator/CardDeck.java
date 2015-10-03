package itml.simulator;

import java.util.ArrayList;
import itml.cards.Card;

/**
 *
 *  This class provides the type CardDeck, which represents a deck of playing cards (Card).
 *
 * @author      Yngvi Bjornsson
 *
 * @version     %I%, %G%
 *
 */

public class CardDeck {

    private ArrayList<Card> m_deck;

    /**
     *
     * Constructor, create a new deck.
     *
     */
    public CardDeck( ) {
        m_deck = new ArrayList<Card>();
    }

    /**
     *
     * Add a card to the deck.
     *
     * @param  card           A card to add to the deck.
     *
     */
    public void addCard( Card card ) {
        m_deck.add( card );
    }

    /**
     *
     * Clone the deck
     *
     * @return  New instance of deck.
     *
     */
    public CardDeck clone() {
        CardDeck deck = new CardDeck();
        for ( Card c : m_deck ) {
            deck.addCard( c );
        }
        return deck;
    }

    /**
     *
     * Get cards in deck
     *
     * @return  Cards in deck as a <code>ArrayList</code>.
     *
     */
    public ArrayList<Card> getCards(  ) {
        return m_deck;
    }

    /**
     *
     * Get cards in deck
     *
     * @param  staminaPointsLevel  An integer indicating the stamina point level
     *
     * @return  Cards in deck the agent can play given that its stamina level is <code>staminaPointsLevel</code>.
     *
     */
    public ArrayList<Card> getCards( int staminaPointsLevel ) {
        ArrayList<Card> actions = new ArrayList<Card>();
        for ( Card c : m_deck ) {
            if ( staminaPointsLevel + c.getStaminaPoints() >= 0 ) {
                actions.add( c );
            }
        }
        return actions;
    }

}
