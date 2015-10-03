package itml.simulator;

/**
 *
 *  This class implements the type StateBattle, which keeps track of the current state of a battle.
 *  (i.e., size of arena and information about the two competing agents and their last actions).
 *
 * @author      Yngvi Bjornsson
 *
 * @version     %I%, %G%
 *
 */
import itml.cards.Card;

public class StateBattle {

    final private int m_numColumns;
    final private int m_numRows;
    final private int m_numSteps;

    private int m_stepNumber;
    private StateAgent[] m_stateAgent;
    private Card[] m_lastMove;

    /**
     *
     * Constructor, create a new battle state.
     *
     * @param  numColumns   An integer representing the number of column of the arena.
     * @param  numRows      An integer representing the number of rows of the arena.
     * @param  numSteps     An integer representing the maximum number of steps (turns) a battle can take.
     * @param  stateAgents  An StateAgent array with information about competing agents.
     *
     */
    public StateBattle( int numColumns, int numRows, int numSteps, StateAgent[] stateAgents ) {
        m_numColumns = numColumns;
        m_numRows = numRows;
        m_numSteps = numSteps;
        m_stepNumber = 0;
        m_stateAgent = new StateAgent[stateAgents.length];
        for ( int a=0; a<m_stateAgent.length; ++a ) {
            m_stateAgent[a] = new StateAgent( stateAgents[a] );
        }
        m_lastMove = new Card[stateAgents.length];
    }

    /**
      *
      * Copy Constructor
      *
      * @param  bs     Battle state to copy.
      *
      */
    public StateBattle( StateBattle bs ) {
        m_numColumns = bs.getNumColumns();
        m_numRows = bs.getNumRows();
        m_numSteps = bs.getNumSteps();
        m_stepNumber= bs.getStepNumber();
        m_stateAgent = new StateAgent[bs.m_stateAgent.length];
        for ( int a=0; a<m_stateAgent.length; ++a ) {
            m_stateAgent[a] = new StateAgent( bs.m_stateAgent[a] );
        }
        m_lastMove = new Card[bs.m_lastMove.length];
        for ( int a=0; a<m_lastMove.length; ++a ) {
            m_lastMove[a] = bs.m_lastMove[a];   // No need to clone the cards themselves as they are immutable.
        }
    }

    /**
      *
      * Clone battle state.
      *
      * @return  new cloned instance.
      *
      */
    public Object clone( ) {
        return new StateBattle( this );
    }

    /**
      *
      * Get number of columns of arena.
      *
      * @return  An integer representing number of column
      *
      */
    public int getNumColumns() {
        return m_numColumns;
    }

    /**
      *
      * Get number of rows of arena.
      *
      * @return  An integer representing number of rows
      *
      */
    public int getNumRows() {
        return m_numRows;
    }

    /**
      *
      * Get number of steps (turns) a game can take.
      *
      * @return  An integer representing number of steps
      *
      */
    public int getNumSteps() {
        return m_numSteps;
    }

    /**
      *
      * Get the current number of steps (turns) of the game.
      *
      * @return  An integer representing number of steps
      *
      */
    public int getStepNumber() {
        return m_stepNumber;
    }

    /**
      *
      * Get information about an agent.
      *
      * @param   a  An integer representing the agent number (0 or 1).
      *
      * @return  State of agent <code>a</code>
      *
      */
    public StateAgent getAgentState( int a ) {
        return m_stateAgent[a];
    }

    /**
      *
      * Get information about the last actions each of the agents made.
      *
      * @return  <code>Card</code> array, with last card played by each agent.
      *
      */
    public Card[] getLastMoves() {
        return m_lastMove;
    }

    /**
      *
      * Get string representation of battle state.
      *
      * @return  <code>String</code> representation.
      *
      */
    public String toString() {

        String symbol = "0123456789abcdef";
        StringBuilder text = new StringBuilder();
        for ( int row = this.getNumRows()-1; row >= 0; --row ) {
            for ( int col = 0; col < this.getNumColumns(); ++col ) {
                char c = '.';
                for ( int a = 0; a < m_stateAgent.length; ++a ) {
                    if ( m_stateAgent[a].getCol() == col && m_stateAgent[a].getRow() == row ) {
                        if ( c == '.' ) {
                            c = symbol.charAt(a);
                        }
                        else {
                            c = '*';
                        }
                    }
                }
                text.append( c );
            }
            text.append( "\n" );
        }
        text.append( "step:" + m_stepNumber + " (" + m_numSteps + ")" );
        for ( int a = 0; a < m_stateAgent.length; ++a ) {
           System.out.println( "A: " + a + " stamina: " + m_stateAgent[a].getStaminaPoints()
                                         + " health: " + m_stateAgent[a].getHealthPoints()
                                         + " lastmove: " + ((m_lastMove[a] == null) ? "n/a" : m_lastMove[a].getName() ) );
        }        
        return text.toString();
        
    }


    /**
      *
      * Update battle state in accordance with the cards the agents played.
      *
      * @param  cards An array of cards played by respective agents.
      *
      * @return  <code>true</code> if play resulted in the game finishing, otherwise <code>false</code>.
      *
      */
    public boolean play( Card[] cards ) {

        int defense[] = new int[cards.length];

        for ( int a=0; a<defense.length; ++a ) {
            defense[a] = 0;
        }

        for ( Card.CardActionType type : Card.CardActionType.values() ) {

            for ( int a=0; a < m_stateAgent.length; ++a ) {

                if ( cards[a] == null || cards[a].getType() != type ) continue;

                switch ( type ) {
                case ctMove:
                    int col = m_stateAgent[a].getCol() + cards[a].getCol();
                    col = Math.max( 0, Math.min( m_numColumns-1, col ) );
                    int row = m_stateAgent[a].getRow() + cards[a].getRow();
                    row = Math.max( 0, Math.min( m_numRows-1, row ) );
                    m_stateAgent[a].setCol( col );
                    m_stateAgent[a].setRow( row );
                    m_stateAgent[a].setStaminaPoints( m_stateAgent[a].getStaminaPoints() + cards[a].getStaminaPoints() );
                    break;
                case ctDefend:
                    defense[a] += cards[a].getDefencePoints();
                    m_stateAgent[a].setStaminaPoints( m_stateAgent[a].getStaminaPoints() + cards[a].getStaminaPoints() );
                    break;
                case ctAttack:
                    for ( int oa=0; oa< m_stateAgent.length; ++oa ) {
                        if ( oa != a && (cards[a].getHitPoints() > defense[oa]) &&
                            cards[a].inAttackRange( m_stateAgent[a].getCol(), m_stateAgent[a].getRow(),
                                                    m_stateAgent[oa].getCol(), m_stateAgent[oa].getRow() ) ) {
                            int healthPoints = m_stateAgent[oa].getHealthPoints() - (cards[a].getHitPoints()-defense[oa]);
                            m_stateAgent[oa].setHealthPoints( healthPoints );
                        }
                    }
                    m_stateAgent[a].setStaminaPoints( m_stateAgent[a].getStaminaPoints() + cards[a].getStaminaPoints() );
                    break;
                }
            }

        }

        m_stepNumber++;

        int numAgentsStillStanding = 0;
        for ( int a=0; a < m_stateAgent.length; ++a ) {
            if ( m_stateAgent[a].getHealthPoints() > 0 ) {
                numAgentsStillStanding++;
            }
            m_lastMove[a] = cards[a];
        }

        return numAgentsStillStanding <= 1;
    }

}
