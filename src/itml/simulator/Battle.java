package itml.simulator;

import java.util.ArrayList;
import itml.agents.Agent;
import itml.cards.Card;
import itml.cards.CardRest;

/**
 *
 *  This class provides the type Battle, which takes care of the logistics regarding conducting battles.
 *
 * @author      Yngvi Bjornsson
 *
 * @version     %I%, %G%
 *
 */

public class Battle {

    private int m_numColumns = 0;
    private int m_numRows = 0;
    private CardDeck m_deck;
    private StateAgent[] m_stateAgents;


    /**
    *
    * Constructor, create a battle
    *
    * @param  numColumns     An integer representing the number of columns of the battle arena.
    * @param  numRows        An integer representing the number of rows of the battle arena.
    * @param  deck           A deck of cards that will be used by all agents in the battle.
    * @param  stateAgents    Information about the initial state of the agents that will battle.
    *
    */
    public Battle ( int numColumns, int numRows, CardDeck deck, StateAgent[] stateAgents) {
        m_numColumns = numColumns;
        m_numRows = numRows;
        m_deck = deck;
        m_stateAgents = stateAgents;
    }

    /**
     *
     * Get the number of columns of the battle arena.
     *
     * @return  An integer representing the number of columns.
     *
     */
    public int getNumColumns()
    {
        return m_numColumns;
    }

    /**
     *
     * Get the number of rows of the battle arena.
     *
     * @return  An integer representing the number of rows.
     *
     */
    public int getNumRows()
    {
        return m_numRows;
    }

    /**
     *
     * Get the deck of cards used in the battle.
     *
     * @return  <code>CardDeck</code>
     *
     */
    public CardDeck getDeck()
    {
        return m_deck;
    }


    /**
    *
    * Conduct a battle.
    *
    * @param  doDebug     A boolean flag for controlling debug output.
    * @param  msPerMove   An integer representing the maximum number of milliseconds a move can take.
    * @param  agents      Agents to match against each other.
    * @param  score       An array of double in which the score of each agent will be returned into
    * @param  log         A GameLog, in whicch the progression of the game will be logged into.
    *
    */
    public void run( boolean doDebug, int maxSteps, int msPerMove, Agent[] agents, double [] score, GameLog log ) {


        StateBattle bs = new StateBattle( m_numColumns, m_numRows, maxSteps, m_stateAgents);

        log.clear();
        log.add( (StateBattle) bs.clone() );

        for ( int a=0; a < agents.length; a++ ) {
            long msStart = System.currentTimeMillis();
            agents[a].startGame( a, (StateBattle) bs.clone() );
            long msDuration = System.currentTimeMillis() - msStart;
            if ( msDuration > msPerMove ) {
                System.out.println("WARNING: Agent " + a + " exceeded time limit in startGame("+msDuration+">"+ msPerMove+")");
            }
            score[a] = 0.0;
        }

        Card[] actions = new Card[agents.length];

        boolean isGameOver = false;
        for ( int step=0; step<maxSteps && !isGameOver; ++step ) {

            if ( doDebug ) {
                System.out.println( bs.toString() );
            }
            for ( int a=0; a < agents.length; a++ ) {
                if ( bs.getAgentState( a ).getHealthPoints() > 0 ) {
                    long msStart = System.currentTimeMillis();
                    Card cardAgent = agents[a].act( (StateBattle) bs.clone() );
                    long msDuration = System.currentTimeMillis() - msStart;
                    if ( msDuration > msPerMove ) {
                       System.out.println("WARNING: Agent " + a + " exceeded time limit in act ("+msDuration+">"+ msPerMove+")");
                    }
                    if ( cardAgent == null ) {
                        System.out.println( "Quitting ..." );
                        return;
                    }
                    // Check if action is legal.
                    boolean isLegal = false;
                    StateAgent stateAgent = bs.getAgentState( a );
                    ArrayList<Card> cards = m_deck.getCards( stateAgent.getStaminaPoints() );
                    for ( Card card : cards ) {
                        if ( card.getName().equals(cardAgent.getName()) ) {
                            actions[a] = card;
                            isLegal = true;
                            break;
                        }
                    }
                    if ( !isLegal ) {
                        System.out.println( "Illegitimate action ..." );                        
                        actions[a] = new CardRest();
                    }

                }
                else {
                    actions[a] = null;
                }
            }
            isGameOver = bs.play( actions );
            log.add( (StateBattle) bs.clone() );
        }

        int numAgentsStillStanding = 0;
        for ( int a=0; a < agents.length; a++ ) {
            if ( bs.getAgentState( a ).getHealthPoints() > 0 ) {
                score[a] = 1.0;
                numAgentsStillStanding++;
            }
        }
        if ( numAgentsStillStanding == 0 ) {
            for ( int a=0; a < agents.length; a++ ) {
                 score[a] = 0.5;
             }
        }
        else if ( numAgentsStillStanding > 1 ) {
           for ( int a=0; a < agents.length; a++ ) {
                score[a] /= numAgentsStillStanding;
            }
        }

        for ( int a=0; a < agents.length; a++ ) {
            long msStart = System.currentTimeMillis();
            agents[a].endGame( (StateBattle) bs.clone(), score );
            long msDuration = System.currentTimeMillis() - msStart;
            if ( msDuration > msPerMove ) {
                System.out.println("WARNING: Agent " + a + " exceeded time limit in endGame("+msDuration+">"+ msPerMove+")");
            }
        }
    }

}
