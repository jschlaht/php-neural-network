<?php

namespace App\Tests\Service;

use App\Service\NeuralNetwork;
use PHPUnit\Framework\TestCase;

class NeuralNetworkWeights352Test extends TestCase
{
    private array $weightsInputToHiddden = [
        [0.9, 0.3, 0.4],
        [0.2, 0.8, 0.2],
        [0.1, 0.5, 0.6],
        [0.6, 0.3, 0.8],
        [0.7, 0.1, 0.3],
    ];
    private array $weightsHidddenToOutput = [
        [0.3, 0.7, 0.8, 0.5, 0.4],
        [0.6, 0.5, 0.2, 0.1, 0.9],
    ];

    private array $expectedHiddenInputs = [];
    private array $expectedHiddenOutputs = [];
    private array $expectedFinalInputs = [];
    private array $expectedFinalOutputs = [];
    private array $expectedOutputsError = [];
    private array $expectedHiddenError = [];
    private int $inputNodes = 3;
    private int $hiddenNodes = 5;
    private int $outputNodes = 2;

    public function testTrain()
    {
        $neuralNetwork = new NeuralNetwork($this->inputNodes, $this->hiddenNodes, $this->outputNodes);

        $inputList = array(0.9, 0.1, 0.8);
        $targetList = array(0.8, 0.6);
        $neuralNetwork->train($inputList, $targetList);
    }

    public function testQuery()
    {

    }
}
