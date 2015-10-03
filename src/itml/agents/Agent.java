package itml.agents;

import itml.cards.Card;
import itml.simulator.CardDeck;
import itml.simulator.StateBattle;
import weka.classifiers.Classifier;
import weka.core.Instances;

/**
 *
 *  This class provides the base Agent type, from which other agents should subclass.
 *
 * @author      Yngvi Bjornsson
 *
 * @version     %I%, %G%
 *
 */
public abstract class Agent {

    protected CardDeck     m_deck;         // The deck of cards to choose from.
    protected int          m_msConstruct;  // Maximum time you can use in the constructor.
    protected int          m_msPerMove;    // Maximum time you can use per act, startGame, endGame call.
    protected int          m_msLearn;      // Maximum time you can use in the learn() method.

    /**
     *
     * Constructor, create a new agent using a given deck of cards.
     *
     * @param  deck           The deck of cards the agent can use.
     * @param  msConstruct    The maximum time (in milliseconds) the constructor can take.
     * @param  msPerMove      The maximum time (in milliseconds) the agent can spend on each individual action (act call).
     * @param  msLearn        The maximum time (in milliseconds) the agent can spend for learning (learn call).
     *
     */
     public Agent( CardDeck deck, int msConstruct, int msPerMove, int msLearn ) {
        m_deck = deck;
        m_msConstruct = msConstruct;
        m_msPerMove = msPerMove;
        m_msLearn = msLearn;
    }

    /**
     * Start a new game. Do any book keeping you need to to at the beginning of a game here.
     * The index of your agent in stateBattle is provided; it will stay unchanged through the
     * entire game, however, the state of the battle may change each step.
     *
     * @param  noThisAgent    The index of the agent in the <code>stateBattle</code> state.
     * @param  stateBattle    The initial battle state.
     *
     */
    public abstract void startGame( int noThisAgent, StateBattle stateBattle );

    /**
     * Game finished. Results is the outcome of the game.
     * entire game, however, the state of the battle may change each step.
     *
     * @param  stateBattle   The initial battle state.
     * @param  results       A double array with the game result for each agent (0.0=loss, 0.5=tie, 1.0=win).
     *
     */
    public abstract void endGame( StateBattle stateBattle, double [] results );

    /**
     * Make your move, by picking a card to play.
     * The index of your agent in stateBattle is provided; it will stay unchanged through the
     * entire game, however, the state of the battle may change each step.
     *
     * @param  stateBattle   The current battle (game) state.
     *
     * @return <code>Card</code> The card to play (action to take).
     *
     */
    public abstract Card act( StateBattle stateBattle );

    /**
     * The agent gets information about past matches of the opponent, and can thus (potentially) learn
     * useful strategies against it for upcoming match. The learning cannot take more than time indicated
     * in the constructor.
     *
     * @param  instances  WEKA instances of past experience.
     *                    (you can look at file "history.arff" to see attribute formats).
     * @return A WEKA classifier, which predicts opponent's move in a given situation.
     *
     */
    public abstract Classifier learn( Instances instances );

}
