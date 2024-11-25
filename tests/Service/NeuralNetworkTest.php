<?php

namespace App\Tests\Service;

use App\Service\NeuralNetwork;
use PHPUnit\Framework\TestCase;

class NeuralNetworkTest extends TestCase
{
    public function testNeuralNetwork() {
        $neuralNetwork = new NeuralNetwork();
        $this->assertInstanceOf(NeuralNetwork::class, $neuralNetwork);
    }


}
